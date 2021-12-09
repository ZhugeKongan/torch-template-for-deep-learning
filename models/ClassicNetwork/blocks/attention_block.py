import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device=torch.device('cuda')

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class STN(nn.Module):
    """Constructs a ECA module.
    Args:
        inplanes: Number of channels of the input feature map

    """
    def __init__(self,in_planes, places):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_planes, places, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(places, places, kernel_size=5),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size()[0], -1)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class softmax_layer(nn.Module):
    """Constructs a ECA module.
        Args:
            input: [B,K,F]
           output: [B,F]
        """
    def __init__(self, dim=512):
        super(softmax_layer, self).__init__()

        self.dim = dim
        self.w_omega = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x):
        # inputs的形状是[B,K,F]
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是[B,K,F]
        att = torch.matmul(u, self.u_omega)
        # att形状是[B,K,1]
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为[B,K,1]
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        outs = torch.sum(scored_x, dim=1)
        return outs


class SE(nn.Module):
    '''
    input: (bt_size, C, H, W)
    output:
        alpha:(C,1)
        att_output:(bt_size, C, H, W)
    '''
    def __init__(self, inplanes):
        super(SE, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if (inplanes%16)==0:
            self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
            self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        else:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1)
            self.conv2 = nn.Conv2d(inplanes , inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)
        # print(out)
        return x * out

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

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
        self.selayer = SE(planes*self.expansion)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(out.size())
        # print(out[0][0][0:3])
        out = self.selayer(out)
        # print(out[0][0][0:3])
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

class residual_attention(nn.Module):
    def __init__(self,block, in_channels, out_channels, size1, size2, size3):
        super(residual_attention, self).__init__()

        self.first_residual_blocks = block(in_channels, out_channels)
        #**trunk_branch**
        self.trunk_branches = nn.Sequential(
            block(in_channels, out_channels),
            block(in_channels, out_channels)
        )
        # **mask_branch**
        #down
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = block(in_channels, out_channels)
        self.skip1_connection_residual_block = block(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = block(in_channels, out_channels)
        self.skip2_connection_residual_block = block(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            block(in_channels, out_channels),
            block(in_channels, out_channels)
        )
        #up
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = block(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = block(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        # attention weght
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        #v3
        # self.atten = nn.Conv1d(in_channels= 2, out_channels=1, kernel_size=1)
        self.w1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

        self.last_blocks = block(in_channels, out_channels)

    def forward(self, x):
        # print(x.size())  # [128, 64, 30, 30]
        x = self.first_residual_blocks(x)
        # print(x.size())#[128, 64, 30, 30]
        out_trunk = self.trunk_branches(x)
        # print(x.size())
        out_mpool1 = self.mpool1(x)
        # print(out_mpool1.size())#[128, 64, 15, 15]
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        # print(out_mpool2.size())#[128, 64, 8, 8]
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        # print(out_mpool3.size())#[128, 64, 4, 4]
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        # print(out_softmax3.size())#[128, 64, 4, 4]
        #
        out_interp3 = self.interpolation3(out_softmax3)
        # print(out_interp3.size())#[128, 64, 15, 15]
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)

        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = out_softmax6 * out_trunk

        out=self.w1*out+self.w2*out_trunk
        # print(self.w1,self.w2)
        out_last = self.last_blocks(out)

        return out_last
class residual_attention(nn.Module):
    def __init__(self,block, in_channels, out_channels, size1, size2, size3):
        super(residual_attention, self).__init__()

        self.first_residual_blocks = block(in_channels, out_channels)
        #**trunk_branch**
        self.trunk_branches = nn.Sequential(
            block(in_channels, out_channels),
            block(in_channels, out_channels)
        )
        # **mask_branch**
        #down
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = block(in_channels, out_channels)
        self.skip1_connection_residual_block = block(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = block(in_channels, out_channels)
        self.skip2_connection_residual_block = block(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            block(in_channels, out_channels),
            block(in_channels, out_channels)
        )
        #up
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = block(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = block(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        # attention weght
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        #v3
        # self.atten = nn.Conv1d(in_channels= 2, out_channels=1, kernel_size=1)
        self.w1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

        self.last_blocks = block(in_channels, out_channels)

    def forward(self, x):
        # print(x.size())  # [128, 64, 30, 30]
        x = self.first_residual_blocks(x)
        # print(x.size())#[128, 64, 30, 30]
        out_trunk = self.trunk_branches(x)
        # print(x.size())
        out_mpool1 = self.mpool1(x)
        # print(out_mpool1.size())#[128, 64, 15, 15]
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        # print(out_mpool2.size())#[128, 64, 8, 8]
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        # print(out_mpool3.size())#[128, 64, 4, 4]
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        # print(out_softmax3.size())#[128, 64, 4, 4]
        #
        out_interp3 = self.interpolation3(out_softmax3)
        # print(out_interp3.size())#[128, 64, 15, 15]
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)

        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = out_softmax6 * out_trunk

        out=self.w1*out+self.w2*out_trunk
        # print(self.w1,self.w2)
        out_last = self.last_blocks(out)

        return out_last

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet, self).__init__()
        self.K = 5

        self.in_planes = 64

        self.conv1 = Conv1(in_planes=3, places=64)
        self.stn=STN(in_planes=3, places=64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc1 = nn.Linear(512,256 )
        self.fc2 = nn.Linear(256,256)
        self.classifer1 = nn.Linear(512, num_classes)
        self.classifer2 = nn.Linear(256, num_classes)

        self.w = Variable(torch.zeros(512, 512).cuda())
        self.b = Variable(torch.zeros(1, 512).cuda())
        self.u = Variable(torch.zeros(1, 512).cuda())

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def attention_net(self, cnn_output):
        alpha=[]
        # print(cnn_output.size())#[4, 2048]
        for i in range(cnn_output.size()[0]):
            x=torch.unsqueeze(cnn_output[i],0)
            x = torch.mm(x,self.w) + self.b
            x = torch.mm(x,self.u.t())
            alpha.append(x)
        return alpha

    def forward(self, x):
        plt.imshow(x[0].permute(1,2,0))
        plt.show()
        x=self.stn(x)
        plt.imshow(x[0].permute(1, 2, 0))
        plt.show()
        print('1:', x.shape)
        x = self.conv1(x)
        # print(x.size())
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())#([32, 512, 4, 4])
        x = self.avgpool(x)
        # print(x.size())#([32, 512, 1, 1])
        x = x.view(x.size()[0], -1)  # 4, 2048
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifer2(x)

        # print(x1.size(),x2.size())#[4, 1],[4, 1]
        return torch.sigmoid(x)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck,[3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck,[3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck,[3, 8, 36, 3])
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())




