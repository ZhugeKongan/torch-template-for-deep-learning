import torch
from torch import nn
from torch.nn import functional as F

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kelnel_size=3,stride=1,padding=0,bias=False):
        super(Conv2dBnRelu,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kelnel_size,stride=stride,padding=padding,bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool1d(5)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.conv1 = nn.Conv1d(7, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_all =self.shared_MLP(self.avg_pool(x))
        max_all =self.shared_MLP(self.max_pool(x))
        B, C, H, W = x.size()
        sortx = x.view(B, C, -1)
        sortx = torch.sort(sortx, dim=-1)[0].clone()  # .detach().requires_grad_(True)#
        bavg_out = self.global_avgpool(sortx)  # B,C,5
        avg_out = torch.cat([avg_all, max_all], dim=1).view(B,2,-1)
        for i in range(5):
            m=bavg_out[:,:,i].unsqueeze(-1).unsqueeze(-1)#B,C,1,1
            avg_m=self.shared_MLP(m).view(B, 1, -1)#B,1,C
            avg_out=torch.cat([avg_out,avg_m],dim=1)
        # print(avg_out.size())##[128, 7, 64]
        out = self.conv1(avg_out).view(B, -1, 1)
        out = out.unsqueeze(-1)
        out = self.sigmoid(out)
        # print(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self,in_planes):
        super(SpatialAttention, self).__init__()

        #Pyramid attention
        self.pyramid1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(in_planes, 1, kelnel_size=7, stride=1, padding=3),
        )
        self.pyramid2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2),
        )
        self.pyramid3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1),
        )
        self.conv1 = Conv2dBnRelu(1, 1, kelnel_size=7, stride=1, padding=3)
        self.conv2 = Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h,w=x.size(2),x.size(3)

        x1 = self.pyramid1(x)
        x2 = self.pyramid2(x1)
        x3 = self.pyramid3(x2)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)
        x = x3 + x2
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        return self.sigmoid(x)


class MFPA(nn.Module):
    def __init__(self, planes):
        super(MFPA, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention(planes)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

if __name__ == '__main__':
    img = torch.randn(16, 32,120, 120)
    net = MFPA(32)
    print(net)
    out = net(img)
    print(out.size())