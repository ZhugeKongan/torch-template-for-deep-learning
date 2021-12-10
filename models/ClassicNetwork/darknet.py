# -*- coding: UTF-8 -*-
"""
An unofficial implementation of Darknet with pytorch
@Cai Yichao 2020_09_08
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.blocks.conv_bn import BN_Conv2d
from models.blocks.residual_blocks import Dark_block


class DarkNet(nn.Module):

    def __init__(self, layers: object, num_classes, is_se=False) -> object:
        super(DarkNet, self).__init__()
        self.is_se = is_se
        filters = [64, 128, 256, 512, 1024]

        self.conv1 = BN_Conv2d(3, 32, 3, 1, 1)
        self.redu1 = BN_Conv2d(32, 64, 3, 2, 1)
        self.conv2 = self.__make_layers(filters[0], layers[0])
        self.redu2 = BN_Conv2d(filters[0], filters[1], 3, 2, 1)
        self.conv3 = self.__make_layers(filters[1], layers[1])
        self.redu3 = BN_Conv2d(filters[1], filters[2], 3, 2, 1)
        self.conv4 = self.__make_layers(filters[2], layers[2])
        self.redu4 = BN_Conv2d(filters[2], filters[3], 3, 2, 1)
        self.conv5 = self.__make_layers(filters[3], layers[3])
        self.redu5 = BN_Conv2d(filters[3], filters[4], 3, 2, 1)
        self.conv6 = self.__make_layers(filters[4], layers[4])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[4], num_classes)

    def __make_layers(self, num_filter, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(Dark_block(num_filter, self.is_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.redu1(out)
        out = self.conv2(out)
        out = self.redu2(out)
        out = self.conv3(out)
        out = self.redu3(out)
        out = self.conv4(out)
        out = self.redu4(out)
        out = self.conv5(out)
        out = self.redu5(out)
        out = self.conv6(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # return F.softmax(out)
        return out


def darknet_53(num_classes=1000):
    return DarkNet([1, 2, 8, 8, 4], num_classes)


# def test():
#     net = darknet_53()
#     summary(net, (3, 256, 256))
#
# test()
