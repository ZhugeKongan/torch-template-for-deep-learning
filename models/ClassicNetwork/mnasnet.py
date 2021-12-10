# -*- coding: UTF-8 -*-
"""
@Cai Yichao 2020_10_29
"""
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.SE_block import SE
from models.blocks.conv_bn import BN_Conv2d


class SeqConv(nn.Module):

    def __init__(self, in_chnls, out_chnls, kernel_size, activation=nn.ReLU(inplace=True)):
        super(SeqConv, self).__init__()
        self.DWConv = BN_Conv2d(in_chnls, in_chnls, kernel_size, stride=1,
                                padding=kernel_size//2, groups=in_chnls, activation=activation)
        self.trans = BN_Conv2d(in_chnls, out_chnls, 1, 1, 0, activation=None)   # Linear activation

    def forward(self, x):
        out = self.DWConv(x)
        return self.trans(out)


class MBConv(nn.Module):
    """Mobile inverted bottleneck conv"""

    def __init__(self, in_chnls, out_chnls, kernel_size, expansion, stride, is_se = False,
                 activation=nn.ReLU(inplace=True)):
        super(MBConv, self).__init__()
        self.is_se = is_se
        self.is_shortcut = (stride == 1) and (in_chnls == out_chnls)
        self.trans1 = BN_Conv2d(in_chnls, in_chnls*expansion, 1, 1, 0, activation=activation)
        self.DWConv = BN_Conv2d(in_chnls*expansion, in_chnls*expansion, kernel_size, stride=stride,
                                padding=kernel_size//2, groups=in_chnls*expansion, activation=activation)
        if self.is_se:
            self.se = SE(in_chnls*expansion, 4)  #se ratio = 0.25
        self.trans2 = BN_Conv2d(in_chnls*expansion, out_chnls, 1, 1, 0, activation=None)    # Linear activation

    def forward(self, x):
        out = self.trans1(x)
        out = self.DWConv(out)
        if self.is_se:
            coeff = self.se(out)
            out *= coeff
        out = self.trans2(out)
        if self.is_shortcut:
            out += x
        return out


class MnasNet_A1(nn.Module):
    """MnasNet-A1"""

    _defaults = {
        "blocks": [2, 3, 4, 2, 3, 1],
        "chnls": [24, 40, 80, 112, 160, 320],
        "expans": [6, 3, 6, 6, 6, 6],
        "k_sizes": [3, 5, 3, 3, 5, 3],
        "strides": [2, 2, 2, 1, 2, 1],
        "is_se": [False, True, False, True, True, False],
        "dropout_ratio": 0.2
    }

    def __init__(self, num_classes = 1000, input_size=224):
        super(MnasNet_A1, self).__init__()
        self.__dict__.update(self._defaults)
        self.body = self.__make_body()
        self.trans = BN_Conv2d(self.chnls[-1], 1280, 1, 1, 0, activation=nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(self.dropout_ratio), nn.Linear(1280, num_classes))

    def __make_block(self, id):
        in_chnls = 16 if id == 0 else self.chnls[id-1]
        strides = [self.strides[id]] + [1] * (self.blocks[id] - 1)
        layers = []
        for i in range(self.blocks[id]):
            layers.append(MBConv(in_chnls, self.chnls[id], self.k_sizes[id], self.expans[id], strides[i],
                                 self.is_se[id]))
            in_chnls = self.chnls[id]
        return nn.Sequential(*layers)

    def __make_body(self):
        blocks = [BN_Conv2d(3, 32, 3, 2, 1, activation=None), SeqConv(32, 16, 3)]
        for index in range(len(self.blocks)):
            blocks.append(self.__make_block(index))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.body(x)
        out = self.trans(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        # return F.softmax(out)
        return out










