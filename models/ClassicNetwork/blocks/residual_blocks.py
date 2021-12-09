# -*- coding: UTF-8 -*-
"""
@Cai Yichao 2020_09_08
"""
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.SE_block import SE
from models.blocks.conv_bn import BN_Conv2d, BN_Conv2d_Leaky


class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    message = "basic"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=None)
        if self.is_se:
            self.se = SE(out_channels, 16)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return F.relu(out)


class BottleNeck(nn.Module):
    """
    BottleNeck block for RestNet-50, ResNet-101, ResNet-152
    """
    message = "bottleneck"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=None)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)

        # fit input with residual output
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        return F.relu(out)


class Dark_block(nn.Module):
    """block for darknet"""
    def __init__(self, channels, is_se=False, inner_channels=None):
        super(Dark_block, self).__init__()
        self.is_se = is_se
        if inner_channels is None:
            inner_channels = channels // 2
        self.conv1 = BN_Conv2d_Leaky(channels, inner_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(inner_channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        if self.is_se:
            self.se = SE(channels, 16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += x
        return F.leaky_relu(out)
