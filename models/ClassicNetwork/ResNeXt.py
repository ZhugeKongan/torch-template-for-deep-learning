# -*- coding: UTF-8 -*-
"""
An unofficial implementation of ResNeXt with pytorch
@Cai Yichao 2020_09_14
"""
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks.conv_bn import BN_Conv2d
from models.blocks.resnext_block import ResNeXt_Block


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: object, cardinality, group_depth, num_classes, is_se=False) -> object:
        super(ResNeXt, self).__init__()
        self.is_se = is_se
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv2d(3, self.channels, 3, stride=1, padding=1)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.___make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.___make_layers(d4, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels, num_classes)  # 224x224 input size

    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride, self.is_se))
            self.channels = self.cardinality * d * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = F.softmax(self.fc(out))
        out = self.fc(out)
        return out


def resNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes)


def resNeXt101_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 32, 4, num_classes)


def resNeXt101_64x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 64, 4, num_classes)


def resNeXt50_32x4d_SE(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes, is_se=True)

import torch
if __name__ == '__main__':
    net = resNeXt50_32x4d(num_classes=1000)
    y= net(torch.randn(10, 3, 64, 64))
    # print(net)
    print(y.size())
    from thop import profile
    input = torch.randn(1, 3, 64, 64)
    flops, params = profile(net, inputs=(input, ))
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    print('  + Number of params: %.3fG' % (flops / 1e9))
    print('flops: ', flops, 'params: ', params)