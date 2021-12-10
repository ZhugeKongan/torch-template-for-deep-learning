# -*- coding: UTF-8 -*-
"""
An unofficial implementation of DPN with pytorch
@Cai Yichao 2020_09_16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.conv_bn import BN_Conv2d
from models.blocks.dpn_block import DPN_Block


class DPN(nn.Module):
    def __init__(self, blocks: object, add_chnls: object, cat_chnls: object,
                 conv1_chnl, cardinality, d, num_classes) -> object:
        super(DPN, self).__init__()
        self.cdty = cardinality
        self.chnl = conv1_chnl
        self.conv1 = BN_Conv2d(3, self.chnl, 7, 2, 3)
        d1 = d
        self.conv2 = self.__make_layers(blocks[0], add_chnls[0], cat_chnls[0], d1, 1)
        d2 = 2 * d1
        self.conv3 = self.__make_layers(blocks[1], add_chnls[1], cat_chnls[1], d2, 2)
        d3 = 2 * d2
        self.conv4 = self.__make_layers(blocks[2], add_chnls[2], cat_chnls[2], d3, 2)
        d4 = 2 * d3
        self.conv5 = self.__make_layers(blocks[3], add_chnls[3], cat_chnls[3], d4, 2)
        self.fc = nn.Linear(self.chnl, num_classes)

    def __make_layers(self, block, add_chnl, cat_chnl, d, stride):
        layers = []
        strides = [stride] + [1] * (block - 1)
        for i, s in enumerate(strides):
            layers.append(DPN_Block(self.chnl, add_chnl, cat_chnl, self.cdty, d, s))
            self.chnl = add_chnl + cat_chnl
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # return F.softmax(out)
        return out


def dpn_92_32x3d(num_classes=1000):
    return DPN(blocks=[3, 4, 20, 3],
               add_chnls=[256, 512, 1024, 2048],
               cat_chnls=[16, 32, 24, 128],
               conv1_chnl=64,
               cardinality=32,
               d=3,
               num_classes=num_classes)


def dpn_98_40x4d(num_classes=1000):
    return DPN(blocks=[3, 6, 20, 3],
               add_chnls=[256, 512, 1024, 2048],
               cat_chnls=[16, 32, 32, 128],
               conv1_chnl=96,
               cardinality=40,
               d=5,
               num_classes=num_classes)


def dpn_131_40_4d(num_classes=1000):
    return DPN(blocks=[4, 8, 28, 3],
               add_chnls=[256, 512, 1024, 2048],
               cat_chnls=[16, 32, 32, 128],
               conv1_chnl=128,
               cardinality=40,
               d=5,
               num_classes=num_classes)

# def test():
#     # net = dpn_92_32x3d()
#     net = dpn_98_40x4d()
#     # net = dpn_131_40_4d()
#     summary(net, (3, 224, 224))
#
# test()
