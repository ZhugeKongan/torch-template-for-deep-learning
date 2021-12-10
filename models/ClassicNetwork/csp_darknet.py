# -*- coding: UTF-8 -*-
"""
An unofficial implementation of CSP-DarkNet with pytorch
@Cai Yichao 2020_09_30
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.blocks.conv_bn import Mish, BN_Conv_Mish


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, chnls, inner_chnnls=None):
        super(ResidualBlock, self).__init__()
        if inner_chnnls is None:
            inner_chnnls = chnls
        self.conv1 = BN_Conv_Mish(chnls, inner_chnnls, 1, 1, 0)     # always use samepadding
        self.conv2 = nn.Conv2d(inner_chnnls, chnls, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(chnls)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out) + x
        return Mish()(out)


class CSPFirst(nn.Module):
    """
    First CSP Stage
    """
    def __init__(self, in_chnnls, out_chnls):
        super(CSPFirst, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnnls, out_chnls, 3, 2, 1)     # same padding
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)
        self.block = ResidualBlock(out_chnls, out_chnls//2)
        self.trans_cat = BN_Conv_Mish(2*out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.block(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSPStem(nn.Module):
    """
    CSP structures including downsampling
    """
    
    def __init__(self, in_chnls, out_chnls, num_block):
        super(CSPStem, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnls, out_chnls, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls//2, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls//2, 1, 1, 0)
        self.blocks = nn.Sequential(*[ResidualBlock(out_chnls//2) for _ in range(num_block)])
        self.trans_cat = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.blocks(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSP_DarkNet(nn.Module):
    """
    CSP-DarkNet
    """

    def __init__(self, num_blocks: object, num_classes=1000) -> object:
        super(CSP_DarkNet, self).__init__()
        chnls = [64, 128, 256, 512, 1024]
        self.conv0 = BN_Conv_Mish(3, 32, 3, 1, 1)   # same padding
        self.neck = CSPFirst(32, chnls[0])
        self.body = nn.Sequential(
            *[CSPStem(chnls[i], chnls[i+1], num_blocks[i]) for i in range(4)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(chnls[4], num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.neck(out)
        out = self.body(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # return F.softmax(out)
        return out


def csp_darknet_53(num_classes=1000):
    return CSP_DarkNet([2, 8, 8, 4], num_classes)

