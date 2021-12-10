# -*- coding: UTF-8 -*-
"""
An unofficial implementation of GoogleNet with pytorch
without Auxiliary classifier
@Cai Yichao 2020_09_09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.conv_bn import BN_Conv2d


class Inception_builder(nn.Module):
    """
    types of Inception block
    """

    def __init__(self, block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super(Inception_builder, self).__init__()
        self.block_type = block_type  # controlled by strings "type1", "type2"

        # 5x5 reduce, 5x5
        self.branch1_type1 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False),
            BN_Conv2d(b1_reduce, b1, 5, stride=1, padding=2, bias=False)  # same padding
        )

        # 5x5 reduce, 2x3x3
        self.branch1_type2 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False),
            BN_Conv2d(b1_reduce, b1, 3, stride=1, padding=1, bias=False),  # same padding
            BN_Conv2d(b1, b1, 3, stride=1, padding=1, bias=False)
        )

        # 3x3 reduce, 3x3
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_reduce, 1, stride=1, padding=0, bias=False),
            BN_Conv2d(b2_reduce, b2, 3, stride=1, padding=1, bias=False)
        )

        # max pool, pool proj
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),  # to keep size, also use same padding
            BN_Conv2d(in_channels, b3, 1, stride=1, padding=0, bias=False)
        )

        # 1x1
        self.branch4 = BN_Conv2d(in_channels, b4, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if self.block_type == "type1":
            out1 = self.branch1_type1(x)
            out2 = self.branch2(x)
        elif self.block_type == "type2":
            out1 = self.branch1_type2(x)
            out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)

        return out


class GoogleNet(nn.Module):
    """
    Inception-v1, Inception-v2
    """

    def __init__(self, str_version, num_classes):
        super(GoogleNet, self).__init__()
        self.block_type = "type1" if str_version == "v1" else "type2"
        self.version = str_version  # "v1", "v2"
        self.conv1 = BN_Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.conv2 = BN_Conv2d(64, 192, 3, stride=1, padding=1, bias=False)
        self.inception3_a = Inception_builder(self.block_type, 192, 16, 32, 96, 128, 32, 64)
        self.inception3_b = Inception_builder(self.block_type, 256, 32, 96, 128, 192, 64, 128)
        self.inception4_a = Inception_builder(self.block_type, 480, 16, 48, 96, 208, 64, 192)
        self.inception4_b = Inception_builder(self.block_type, 512, 24, 64, 112, 224, 64, 160)
        self.inception4_c = Inception_builder(self.block_type, 512, 24, 64, 128, 256, 64, 128)
        self.inception4_d = Inception_builder(self.block_type, 512, 32, 64, 144, 288, 64, 112)
        self.inception4_e = Inception_builder(self.block_type, 528, 32, 128, 160, 320, 128, 256)
        self.inception5_a = Inception_builder(self.block_type, 832, 32, 128, 160, 320, 128, 256)
        self.inception5_b = Inception_builder(self.block_type, 832, 48, 128, 192, 384, 128, 384)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception3_a(out)
        out = self.inception3_b(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception4_a(out)
        out = self.inception4_b(out)
        out = self.inception4_c(out)
        out = self.inception4_d(out)
        out = self.inception4_e(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception5_a(out)
        out = self.inception5_b(out)
        out = F.avg_pool2d(out, 7)
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # return F.softmax(out)
        return out


# class Inception_v3(nn.Module):
#     def __init__(self):


def inception_v1():
    return GoogleNet("v1", num_classes=1000)


def inception_v2():
    return GoogleNet("v2", num_classes=1000)

# def test():
#     net = inception_v1()
#     # net = inception_v2()
#     summary(net, (3, 224, 224))
#
#
# test()
