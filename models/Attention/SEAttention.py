import numpy as np
import torch
from torch import nn
from torch.nn import init



class SEAttention(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(SEAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SEAttention(channel=512,reduction=8)
    output=se(input)
    print(output.shape)

    