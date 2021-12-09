import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.training=True

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        # print(self.drop_prob)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            # pass
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

# class DropBlock2D(nn.Module):
#     r"""Randomly zeroes spatial blocks of the input tensor.
#     As described in the paper
#     `DropBlock: A regularization method for convolutional networks`_ ,
#     dropping whole blocks of feature map allows to remove semantic
#     information as compared to regular dropout.
#     Args:
#         keep_prob (float, optional): probability of an element to be kept.
#         Authors recommend to linearly decrease this value from 1 to desired
#         value.
#         block_size (int, optional): size of the block. Block size in paper
#         usually equals last feature map dimensions.
#     Shape:
#         - Input: :math:`(N, C, H, W)`
#         - Output: :math:`(N, C, H, W)` (same shape as input)
#     .. _DropBlock: A regularization method for convolutional networks:
#        https://arxiv.org/abs/1810.12890
#     """
#
#     def __init__(self, keep_prob=0.9, block_size=7):
#         super(DropBlock2D, self).__init__()
#         self.keep_prob = keep_prob
#         self.block_size = block_size
#
#     def forward(self, input):
#         if not self.training or self.keep_prob == 1:
#             return input
#         gamma = (1. - self.keep_prob) / self.block_size ** 2
#         for sh in input.shape[2:]:
#             gamma *= sh / (sh - self.block_size + 1)
#         M = torch.bernoulli(torch.ones_like(input) * gamma)
#         Msum = F.conv2d(M,
#                         torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
#                                                                                              dtype=input.dtype),
#                         padding=self.block_size // 2,
#                         groups=input.shape[1])
#         torch.set_printoptions(threshold=5000)
#         mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
#         return input * mask * mask.numel() /mask.sum() #TODO input * mask * self.keep_prob ?

class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)