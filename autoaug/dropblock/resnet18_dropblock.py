import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from augmented.dropblock.scheduler import LinearScheduler
from augmented.autoaug.dropblock.dropblock import DropBlock2D


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=3,stride=1,padding=1, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # print(x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0., block_size=5),
            start_value=0.0,
            stop_value=0.25,
            nr_steps=5e3
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer1 = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.dropblock.step()

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training:
            x = self.dropblock(x)
        x = self.layer4(x)
        if self.training:
            x = self.dropblock(x)

        # print(x.size())  # ([32, 512, 1, 1])
        x = self.avgpool(x)
        # print(x.size())#([32, 512, 1, 1])
        x = x.view(x.size()[0], -1)  # 4, 2048
        # x = self.relu1(self.bn1(self.fc1(x)))
        x1 = self.classifer1(x)
        return x1
def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],**kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3],**kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck,[3, 4, 6, 3],**kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck,[3, 4, 23, 3],**kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck,[3, 8, 36, 3],**kwargs)


if __name__ == '__main__':
    net = ResNet50(num_classes=200)
    y = net(torch.randn(10, 3, 32, 32))
    # print(net)
    print(y.size())
    from thop import profile
    input = torch.randn(1, 3, 64, 64)
    flops, params = profile(net, inputs=(input, ))
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    print('  + Number of params: %.3fG' % (flops / 1e9))
    print('flops: ', flops, 'params: ', params)