import torch
import torch.nn as nn
import torch.nn.functional as F


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
