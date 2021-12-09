# -*- coding:UTF-8 -*-
"""
@Cai Yichao 2020_10_10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.conv_bn import BN_Conv2d, Mish
from models.blocks.SE_block import SE


def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x


class BasicUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.ro_chnls = out_chnls - self.l_chnls
        self.groups = groups

        # layers
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1,  # same padding, depthwise conv
                                 groups=self.ro_chnls, activation=None)
        act = None if self.is_res else nn.ReLU(inplace=True)
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=None)

    def forward(self, x):
        x_l = x[:, :self.l_chnls, :, :]
        x_r = x[:, self.l_chnls:, :, :]

        # right path
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)

        # concatenate
        out = torch.cat((x_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class DSampling(nn.Module):
    """Spatial down sampling of SuffleNet-v2"""

    def __init__(self, in_chnls, groups=2):
        super(DSampling, self).__init__()
        self.groups = groups
        self.dwconv_l1 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1,  # down-sampling, depth-wise conv.
                                   groups=in_chnls, activation=None)
        self.conv_l2 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.conv_r1 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.dwconv_r2 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=None)
        self.conv_r3 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)

    def forward(self, x):
        # left path
        out_l = self.dwconv_l1(x)
        out_l = self.conv_l2(out_l)

        # right path
        out_r = self.conv_r1(x)
        out_r = self.dwconv_r2(out_r)
        out_r = self.conv_r3(out_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        out = shuffle_chnls(out, self.groups)
        return shuffle_chnls(out, self.groups)
