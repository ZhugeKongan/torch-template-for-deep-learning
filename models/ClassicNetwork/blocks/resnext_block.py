# -*- coding: UTF-8 -*-
"""
@Cai Yichao 2020_09_08
"""
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.SE_block import SE
from models.blocks.conv_bn import BN_Conv2d


class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride, is_se=False):
        super(ResNeXt_Block, self).__init__()
        self.is_se = is_se
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls * 2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls * 2)
        if self.is_se:
            self.se = SE(self.group_chnls * 2, 16)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls, self.group_chnls * 2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_chnls * 2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += self.short_cut(x)
        return F.relu(out)
