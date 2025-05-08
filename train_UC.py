import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import Config as config

from dynamic_tanh import convert_ln_to_dyt


dir_checkpoint = Path('./checkpoints/')
train_dir_img = Path('dataset/train/images')
train_dir_mask = Path('dataset/train/masks')
val_dir_img = Path('dataset/test/images')
val_dir_mask = Path('dataset/test/masks')

def set_seed(seed=42):
    """固定训练的随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 确保所有 GPU 都固定相同的 seed
    torch.backends.cudnn.deterministic = True  # 让 cuDNN 以确定的方式运行
    torch.backends.cudnn.benchmark = False     # 禁止 cuDNN 自动优化，保证一致性
def calculate_metrics(pred, target):
    """ 计算 Precision, Recall, Accuracy 和 IOU """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)  # 二值化
    target = target.astype(np.uint8)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    TP = intersection  # 交集
    FP = pred.sum() - TP
    FN = target.sum() - TP
    TN = union - TP - FP - FN

    precision = TP / (TP + FP + 1e-6)  # 避免除零
    recall = TP / (TP + FN + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    iou = TP / (union + 1e-6)

    return precision, recall, accuracy, iou

def train_model(
        model,
        device,
        epochs: int = 2,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        seed: int = 42  # 设定默认的随机种子
):
    # **固定随机种子**
    set_seed(seed)
    # 1. Create dataset
    try:
        train_dataset = CarvanaDataset(train_dir_img, train_dir_mask, img_scale)
        val_dataset = CarvanaDataset(val_dir_img, val_dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        train_dataset = BasicDataset(train_dir_img, train_dir_mask, img_scale)
        val_dataset = BasicDataset(val_dir_img, val_dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 存储最佳结果的变量
    best_val_score = 0
    best_train_score = 0
    best_loss = float('inf')
    best_learning_rate = learning_rate
    best_iou = 0
    best_precision = 0
    best_accuracy = 0


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'网络定义了 {model.n_channels} 输入通道，但加载的图像有 {images.shape[1]} 个通道，请检查图像是否加载正确。'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        dice_score = 1 - dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(),
                                                   multiclass=False)
                        precision, recall, accuracy, iou = calculate_metrics(F.sigmoid(masks_pred.squeeze(1)),
                                                                             true_masks)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        dice_score = 1 - dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        precision, recall, accuracy, iou = calculate_metrics(masks_pred.argmax(dim=1), true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                logging.info(f"Batch {global_step}: Loss={loss.item():.4f}, Dice Score={dice_score.item():.4f}")
                experiment.log({
                    'train loss': loss.item(),
                    'train dice': dice_score.item(),
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'iou': iou,
                    'step': global_step,
                    'epoch': epoch
                })
                # pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.set_postfix(**{'Loss': loss.item(), 'Dice': dice_score.item(),
                                    'Precision': precision, 'Recall': recall,
                                    'Accuracy': accuracy, 'IOU': iou})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        # 保存最佳结果
                        if val_score > best_val_score:
                            best_val_score = val_score
                            best_train_score = dice_score.item()
                            best_loss = loss.item()
                            best_learning_rate = optimizer.param_groups[0]['lr']
                            best_iou = iou
                            best_precision = precision
                            best_accuracy = accuracy

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'step': global_step,
                                'epoch': epoch
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    # 输出最优结果
    logging.info(f"Best Results: "
                 f"Train Score: {best_train_score}, Val Score: {best_val_score}, Loss: {best_loss}, "
                 f"Learning Rate: {best_learning_rate}, IOU: {best_iou}, Precision: {best_precision}, Accuracy: {best_accuracy}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    config_vit = config.get_CTranS_config()
    logging.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    logging.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    logging.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
    model = UNet(config_vit, n_channels=config.n_channels, n_classes=2)
    model = convert_ln_to_dyt(model)
    # model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
