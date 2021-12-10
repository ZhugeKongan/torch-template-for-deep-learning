# -*- coding: UTF-8 -*-
"""
An unofficial implementation of DenseNet with pytorch
@Cai Yichao 2020_09_15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.conv_bn import BN_Conv2d
from models.blocks.dense_block import DenseBlock, CSP_DenseBlock


class DenseNet(nn.Module):

    def __init__(self, layers: object, k, theta, num_classes, part_ratio=0) -> object:
        super(DenseNet, self).__init__()
        # params
        self.layers = layers
        self.k = k
        self.theta = theta
        self.Block = DenseBlock if part_ratio == 0 else CSP_DenseBlock   # 通过part_tatio参数控制block type
        # layers
        self.conv = BN_Conv2d(3, 2 * k, 7, 2, 3)
        self.blocks, patches = self.__make_blocks(2 * k)
        self.fc = nn.Linear(patches, num_classes)

    def __make_transition(self, in_chls):
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(
            BN_Conv2d(in_chls, out_chls, 1, 1, 0),
            nn.AvgPool2d(2)
        ), out_chls

    def __make_blocks(self, k0):
        """
        make block-transition structures
        :param k0:
        :return:
        """
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(self.Block(k0, self.layers[i], self.k))
            patches = k0 + self.layers[i] * self.k  # output feature patches from Dense Block
            if i != len(self.layers) - 1:
                transition, k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list), patches

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, 3, 2, 1)
        # print(out.shape)
        out = self.blocks(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 7)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # out = F.softmax(self.fc(out))
        out = self.fc(out)
        return out


def densenet_121(num_classes=1000):
    return DenseNet([6, 12, 24, 16], k=32, theta=0.5, num_classes=num_classes)


def densenet_169(num_classes=1000):
    return DenseNet([6, 12, 32, 32], k=32, theta=0.5, num_classes=num_classes)


def densenet_201(num_classes=1000):
    return DenseNet([6, 12, 48, 32], k=32, theta=0.5, num_classes=num_classes)


def densenet_264(num_classes=1000):
    return DenseNet([6, 12, 64, 48], k=32, theta=0.5, num_classes=num_classes)


def csp_densenet_121(num_classes=1000):
    return DenseNet([6, 12, 24, 16], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


def csp_densenet_169(num_classes=1000):
    return DenseNet([6, 12, 32, 32], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


def csp_densenet_201(num_classes=1000):
    return DenseNet([6, 12, 48, 32], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


def csp_densenet_264(num_classes=1000):
    return DenseNet([6, 12, 64, 48], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


# def test():
#     net = densenet_264()
#     summary(net, (3, 224, 224))
#     x = torch.randn((2, 3, 224, 224))
#     y = net(x)
#     print(y.shape)
#
#
# test()
