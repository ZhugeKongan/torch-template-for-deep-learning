#特征金字塔注意力模型
#https://github.com/xgmiao/Pyramid-Attention-Networks/blob/master/pyramid_attention_network.py
#https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch
import torch
import torch.nn as nn

class BN_Conv2d(nn.Module):
    def __init__(self,in_ch,out_ch,kelnel_size=3,stride=1,padding=0,bias=False):
        super(BN_Conv2d,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kelnel_size,stride=stride,padding=padding,bias=bias),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self,x):
        return self.conv(x)

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

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_ch,out_ch,stride=1):
        super(BasicBlock,self).__init__()

        self.conv1=Conv2dBnRelu(in_ch,out_ch,stride=stride)
        self.conv2=BN_Conv2d(out_ch,out_ch,stride=1)

        self.relu=nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride!=1 or in_ch!=self.expansion*out_ch:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_ch,self.expansion*out_ch,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion *out_ch)
            )
    def forward(self,x):
        residual=x

        x=self.conv1(x)
        x=self.conv2(x)
        # attention
        x+=self.shortcut(residual)
        x=self.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_ch,out_ch,stride=1):
        super(Bottleneck,self).__init__()

        self.conv1=Conv2dBnRelu(in_ch,out_ch,kelnel_size=1,stride=1,bias=False)
        self.conv2=Conv2dBnRelu(out_ch,out_ch,kelnel_size=3,stride=stride,padding=1,bias=False)
        self.conv3=BN_Conv2d(out_ch,out_ch*self.expansion,kelnel_size=1,stride=1,bias=False)

        self.relu=nn.ReLU(inplace=True)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_ch!=self.expansion*out_ch:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_ch,out_ch*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_ch*self.expansion)
            )
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        #attention
        out+=self.shortcut(x)
        out=self.relu(out)

        return out

'''*******************************
Feature Pyramid Attention Module
FPAModule1:
	downsample use maxpooling
*******************************'''
class FPAModule1(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(FPAModule1,self).__init__()

        #global pooling branch
        self.branch1=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch,out_ch,kelnel_size=1,stride=1,padding=0),
        )
        #middle branch
        self.mid=nn.Sequential(
            Conv2dBnRelu(in_ch,out_ch,kelnel_size=1,stride=1,padding=0)
        )
        #Pyramid branch
        self.pyramid1=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            Conv2dBnRelu(in_ch,1,kelnel_size=7,stride=1,padding=3),
        )
        self.pyramid2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2),
        )
        self.pyramid3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1),
        )
        self.conv1=Conv2dBnRelu(1, 1, kelnel_size=7, stride=1, padding=3)
        self.conv2 = Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1)

    def forward(self,x):
        h,w=x.size(2),x.size(3)
        b1=self.branch1(x)
        b1=nn.Upsample(size=(h,w),mode='bilinear',align_corners=True)(b1)

        mid=self.mid(x)

        x1=self.pyramid1(x )
        x2=self.pyramid2(x1)
        x3=self.pyramid3(x2)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x3 = nn.Upsample(size=(h // 4, w // 4),mode='bilinear',align_corners=True )(x3)
        x=x3+x2
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        x=x+x1
        x = nn.Upsample(size=(h , w ), mode='bilinear', align_corners=True)(x)

        x=torch.mul(x,mid)
        x+=b1

        return  x

'''
Feature Pyramid Attention Module
FPAModule2:
	downsample use convolution with stride = 2
'''
class FPAModule2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(FPAModule2,self).__init__()

        #global pooling branch
        self.branch1=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch,out_ch,kelnel_size=1,stride=1,padding=0),
        )
        #middle branch
        self.mid=nn.Sequential(
            Conv2dBnRelu(in_ch,out_ch,kelnel_size=1,stride=1,padding=0)
        )
        #Pyramid branch
        self.pyramid1=nn.Sequential(
            Conv2dBnRelu(in_ch,1,kelnel_size=7,stride=2,padding=3),
        )
        self.pyramid2 = nn.Sequential(
            Conv2dBnRelu(1, 1, kelnel_size=5, stride=2, padding=2),
        )
        self.pyramid3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kelnel_size=3, stride=2, padding=1),
        )
        self.conv1=Conv2dBnRelu(1, 1, kelnel_size=7, stride=1, padding=3)
        self.conv2 = Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1)

    def forward(self,x):
        h,w=x.size(2),x.size(3)
        b1=self.branch1(x)
        b1=nn.Upsample(size=(h,w),mode='bilinear',align_corners=True)(b1)

        mid=self.mid(x)

        x1=self.pyramid1(x )
        x2=self.pyramid2(x1)
        x3=self.pyramid3(x2)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x3 = nn.Upsample(size=(h // 4, w // 4),mode='bilinear',align_corners=True )(x3)
        x=x3+x2
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        x=x+x1
        x = nn.Upsample(size=(h , w ), mode='bilinear', align_corners=True)(x)

        x=torch.mul(x,mid)
        x+=b1

        return  x

'''*******************************
 Global Attention Upsample Module
*******************************'''

class GAUModule(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(GAUModule,self).__init__()

        self.low_conv=Conv2dBnRelu(in_ch,out_ch,kelnel_size=3,stride=1,padding=1)
        self.high_conv=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BN_Conv2d(out_ch,out_ch,kelnel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
    def forward(self,lowx,highx):
        h, w = lowx.size(2), lowx.size(3)
        highx = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(highx)
        # print(lowx.size(),highx.size())#[500, 1024, 7, 7],[500, 1024, 7, 7]
        lowx=self.low_conv(lowx)
        h_up=self.high_conv(highx)

        z=lowx*h_up
        # print(highx.size(),z.size())#[500, 1024, 7, 7],[500, 1024, 5, 5]
        return z+highx
'''
papers:
	Pyramid Attention Networks
'''
class PAN(nn.Module):
    def __init__(self,block,num_blocks,num_clssess=43):
        super(PAN,self).__init__()

        self.in_planes=64
        self.conv1=Conv2dBnRelu(in_ch=3,out_ch=64,kelnel_size=7,stride=2,padding=0)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        #分类任务
        # self.fpa=FPAModule2(512*block.expansion,num_clssess)
        # 解码上采样任务
        self.fpa = FPAModule2(512 * block.expansion, num_clssess)
        self.gau1 = GAUModule(512 * block.expansion//2, num_clssess)
        self.gau2 = GAUModule(512 * block.expansion//4, num_clssess)
        self.gau3 = GAUModule(512 * block.expansion//8, num_clssess)



    def _make_layer(self,block,planes,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        h, w = x.size(2), x.size(3)
        x=self.maxpool(self.conv1(x))    #1/4  ([500, 64, 29, 29])
        # print(x.size())
        x1 = self.layer1(x )#1/4
        x2 = self.layer2(x1)#1/8
        x3 = self.layer3(x2)#1/16
        x4 = self.layer4(x3)#1/32#([500, 2048, 4, 4])
        # print(x4.size())
        x5=self.fpa(x4)  #1/16#[500, 1024, 4, 4]
        # print(x5.size())
        x3 = self.gau1(x3, x5)#1/8
        x2 = self.gau2(x2, x3)#1/4
        x1 = self.gau3(x1, x2)#1/2

        out=nn.Upsample(size=(h,w),mode='bilinear', align_corners=True)(x1)#1

        return out

def ResNet18():
    return PAN(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return PAN(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return PAN(Bottleneck,[3, 4, 6, 3])

def ResNet101():
    return PAN(Bottleneck,[3, 4, 23, 3])

def ResNet152():
    return PAN(Bottleneck,[3, 8, 36, 3])


net = ResNet50()
y = net(torch.randn(500, 3, 120, 120))








