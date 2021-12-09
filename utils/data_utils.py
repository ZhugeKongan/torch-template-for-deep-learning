# -*- coding:UTF-8 -*-
"""
data utils for train
"""
# from args import*
import os
import numpy as np
from math import log10
import pandas as pd
import albumentations as albu
from matplotlib import pyplot as plt


import torch
import torchvision
import torchvision.transforms as transforms
# from pytorch_msssim import ssim as calculate_ssim

'''***********- data preprocess-*************'''
def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):
    # print(x.shape)
    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std
    # print("proce",x.shape)
    return x


'''***********- data albumentations-*************'''

#参考：https://blog.csdn.net/zhangyuexiang123/article/details/107705311
def get_training_augmentation(h=256,w=256):
    y = h
    x = w
    # print(x,y)
    # train_transform = [albu.Resize(y, x)]
    # return albu.Compose(train_transform)
    train_transform = [
        albu.Normalize(mean=(123.5, 124.9, 114.8), std=(44.2, 41.7, 39.6), max_pixel_value=1, always_apply=False, p=1.0),#归一化
        albu.RandomRotate90(p=0.5),#随机旋转90
        albu.Flip(p=0.5),#翻转
        albu.Transpose(p=0.5),#变换行列
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),## 将高斯噪声添加到输入图像
            albu.GaussNoise(), # 将高斯噪声应用于输入图像
        ], p=0.2),
        albu.OneOf([
            albu.MotionBlur(p=0.2),# 使用随机大小的内核将运动模糊应用于输入图像
            albu.MedianBlur(blur_limit=3, p=0.1),# 中值滤波
            albu.Blur(blur_limit=3, p=0.1),# 使用随机大小的内核模糊输入图像
        ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),# 随机应用仿射变换：平移，缩放和旋转输入
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),#光学畸变
            albu.GridDistortion(p=0.1),#网格失真
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        albu.OneOf([# 锐化、浮雕等操作
            # albu.CLAHE(clip_limit=2),#直方图均衡化
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),# 随机明亮对比度
        ], p=0.3),
        albu.HueSaturationValue(p=0.3),#色度饱和度
        albu.Resize(y, x,p=1)
        ]
    return albu.Compose(train_transform)

def get_validation_augmentation(h=256,w=256):
    y = h
    x = w
    # print('test')
    """Add paddings to make image shape divisible by 32"""
    test_transform = [#albu.RandomBrightnessContrast(p=0.3),#随机亮度对比度
                       # albu.RandomCrop(400, 400, always_apply=False, p=1),
                       # albu.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
                       albu.Normalize(mean=(123.5, 124.9, 114.8), std=(44.2, 41.7, 39.6), max_pixel_value=1, always_apply=False, p=1.0),#归一化
                       # albu.VerticalFlip(p=0.5),#水平翻转
                       # albu.HorizontalFlip(p=0.5),#垂直翻转
                       # albu.Downscale(p=1.0, scale_min=0.35, scale_max=0.75, ),
                       # albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                       # albu.ChannelShuffle(always_apply=False, p=0.5)
                       #  albu.RandomGamma(gamma_limit=(80, 120), eps=1e-07, always_apply=False, p=0.5)
                       albu.Resize(y, x)

                       ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

'''***********-同 data albumentations-*************'''
def gen_train_loader(path, input_size, batch_size):
    train_set = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomCrop(input_size, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return loader, train_set.class_to_idx


def gen_test_loader(path, input_size, batch_size):
    test_set = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return loader, test_set.class_to_idx
'''***********- 模型保存-*************'''
def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def save_checkpoint(net=None, optimizer=None, epoch=None, train_losses=None, train_acc=None, val_loss=None,
                    val_acc=None, check_loss=None, savepath=None,m_name=None, GPUdevices=1):
    if GPUdevices > 1:
        net_weights = net.module.state_dict()
    else:
        net_weights = net.state_dict()
    save_json = {
        'net_state_dict': net_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    if check_loss > val_loss:
        savepath = savepath + '/{}_best_params.pkl'.format(m_name)
        check_loss = val_loss
    else:
        savepath = savepath + '/{}_epoch_{}.pkl'.format(m_name, epoch)
    torch.save(save_json, savepath)
    print("checkpoint of {}th epoch saved at {}".format(epoch, savepath))

    return check_loss

def load_checkpoint(model = None, optimizer=None, checkpoint_path=None,  losses_flag = None):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['net_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    if not losses_flag:
        return model, optimizer, start_epoch
    else:
        losses = checkpoint['train_losses']
        return model, optimizer, start_epoch, losses

def logger_to_file():
    pass

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s

'''***********- 特征可视化 -*************'''
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(img_batch):
    """Constructs a ECA module.
            Args:
                input: feature[B,H,W,C],img_size
               output: NONE
            """
    feature_map = img_batch[0].detach().cpu().numpy()
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2] #C
    row, col = get_row_col(num_pic) #图片数

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')
        plt.title('feature_map_{}'.format(i))

    plt.savefig('feature_map.png')
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")


def rand_bbox( size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_rat =max(cut_rat,np.sqrt(lam))
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """
    label= np.array([cifar100_dataset[i][1]for i in range(len(cifar100_dataset))])
    # label =cifar100_dataset[:,1]
    print(len(np.where(label==0)[0]),len(np.where(label == 1)[0]),len(np.where(label == 2)[0]),len(np.where(label == 3)[0]))
    # print(len(np.where(label == 1)[0]))
    # print(len(np.where(label == 2)[0]))
    # print(len(np.where(label == 3)[0]))
    # print(len(np.where(label == 4)[0]))
    data_r = np.dstack([cifar100_dataset[i][0][0, :, :] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][0][1, :, :] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][0][2, :, :] for i in range(len(cifar100_dataset))])

    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std
def compute_distribute(cifar100_dataset):
    n=len(cifar100_dataset)
    datas=torch.zeros(100,3,32,32)
    flag=torch.zeros(100,1)
    for i in range(n):
        if flag[cifar100_dataset[i][1]]==1:
            datas[cifar100_dataset[i][1]]=(cifar100_dataset[i][0]+datas[cifar100_dataset[i][1]])/2
        else:
            datas[cifar100_dataset[i][1]]=cifar100_dataset[i][0]
            flag[cifar100_dataset[i][1]] = 1
            print(cifar100_dataset[i][1])
    print(datas.shape)
    return datas


from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:#resnet
        if epoch < 80:
            lr = data_config.lr
        elif epoch < 120:
            lr = data_config.lr * 0.1
        else:
            lr = data_config.lr * 0.01
    elif model_type == 2:#wresnet
        if epoch < 60:
            lr = data_config.lr
        elif epoch < 120:
            lr = data_config.lr * 0.2
        elif epoch < 160:
            lr = data_config.lr * 0.04
        else:
            lr = data_config.lr * 0.008
    elif model_type == 3:#resnext,densenet
        if epoch < 150:
            lr = data_config.lr
        elif epoch < 225:
            lr = data_config.lr * 0.1
        else:
            lr = data_config.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
