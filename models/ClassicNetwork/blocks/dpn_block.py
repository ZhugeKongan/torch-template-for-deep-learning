# -*- coding: UTF-8 -*-
"""
@Cai Yichao 2020_09_16
"""

import torch.nn as nn
import torch.nn.functional as F

from models.blocks.conv_bn import BN_Conv2d


class DPN_Block(nn.Module):
    """
    Dual Path block
    """

    def __init__(self, in_chnls, add_chnl, cat_chnl, cardinality, d, stride):
        super(DPN_Block, self).__init__()
        self.add = add_chnl
        self.cat = cat_chnl
        self.chnl = cardinality * d
        self.conv1 = BN_Conv2d(in_chnls, self.chnl, 1, 1, 0)
        self.conv2 = BN_Conv2d(self.chnl, self.chnl, 3, stride, 1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.chnl, add_chnl + cat_chnl, 1, 1, 0)
        self.bn = nn.BatchNorm2d(add_chnl + cat_chnl)
        self.shortcut = nn.Sequential()
        if add_chnl != in_chnls:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chnls, add_chnl, 1, stride, 0),
                nn.BatchNorm2d(add_chnl)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        add = out[:, :self.add, :, :] + self.shortcut(x)
        out = torch.cat((add, out[:, self.add:, :, :]), dim=1)
        return F.relu(out)
