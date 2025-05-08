# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from .RSDTT import ChannelTransformer
from pytorch_wavelets import DWTForward


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

def get_activation(activation_type):
    """激活函数"""
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()



class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU) 每个卷积层后都跟一个批归一化和激活函数"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)



class DownHWD(nn.Module):
    """下采样模块，使用HWD替代池化"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hwd = HWD(in_channels, out_channels)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.hwd(x)
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    """用于将输入张量展平，使其适应全连接层（线性层）
       假设有一个形状为 [4, 3, 32, 32] 的张量,结果将为[4, 3072]   3072=3*32*32
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.coatt = CS_FFM(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CS_FFM(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)
        kernel_size = 7
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        x = self.relu(x_after_channel)
        # print(out.shape)

        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out1, _ = torch.max(g, dim=1, keepdim=True)
        max_out2, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([max_out1, max_out2], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        x_after_spatial = out * x
        out = self.relu(x_after_spatial)
        return out



class UNet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownHWD(in_channels, in_channels * 2)
        self.down2 = DownHWD(in_channels * 2, in_channels * 4)
        self.down3 = DownHWD(in_channels * 4, in_channels * 8)
        self.down4 = DownHWD(in_channels * 8, in_channels * 8)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels * 16, in_channels * 4)
        self.up3 = UpBlock_attention(in_channels * 8, in_channels * 2)
        self.up2 = UpBlock_attention(in_channels * 4, in_channels)
        self.up1 = UpBlock_attention(in_channels * 2, in_channels)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if nusing BCEWithLogitsLoss or class>1
        if self.vis:  # visualize the attention maps
            return logits, att_weights
        else:
            return logits
