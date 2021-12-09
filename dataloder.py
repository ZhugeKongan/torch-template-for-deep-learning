# -*- coding:UTF-8 -*-
"""
dataset and  data reading
"""

import os
# import glob
# import json
# import functools
import numpy as np
# import pandas as pd
# from osgeo import gdal
# import albumentations as albu
# from skimage.color import gray2rgb
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
#
#
# from utils.arg_utils import *
# from utils.data_utils import *
# from utils.algorithm_utils import *
from autoaug.augmentations import Augmentation
from autoaug.archive import fa_reduced_cifar10,autoaug_paper_cifar10,fa_reduced_imagenet
import autoaug.aug_transforms as aug

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset

from dataset_loder.scoliosis_dataloder import ScoliosisDataset
from autoaug.cutout import Cutout

def training_transforms():
    return transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616]),
            # Cutout()
        #[125.3, 123.0, 113.9],[63.0, 62.1, 66.7]
        ])
def validation_transforms():
    return transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616]),
        ])


def load_dataset(data_config):
    if data_config.dataset == 'cifar10':
        training_transform=training_transforms()
        if data_config.autoaug:
            print('auto Augmentation the data !')
            training_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        train_dataset = torchvision.datasets.CIFAR10(root=data_config.data_path,
                                                     train=True,
                                                     transform=training_transform,
                                                     download=True)
        val_dataset = torchvision.datasets.CIFAR10(root=data_config.data_path,
                                                   train=False,
                                                   transform=validation_transforms(),
                                                   download=True)
        return train_dataset,val_dataset
    elif data_config.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_config.data_path,
                                                     train=True,
                                                     transform=training_transforms(),
                                                     download=True)
        val_dataset = torchvision.datasets.CIFAR100(root=data_config.data_path,
                                                   train=False,
                                                   transform=validation_transforms(),
                                                   download=True)
        return train_dataset, val_dataset

    elif data_config.dataset == 'tiny_imagenet':
        data_path='/disks/disk2/lishengyan/dataset/tiny-imagenet-200'
        traindir = data_path + '/train'
        valdir = data_path + '/val'
        testdir=data_path + '/test'
        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2302, 0.2265, 0.2262])
        train_dataset = torchvision.datasets.ImageFolder(traindir,
                                                         transforms.Compose([
                                                             # transforms.RandomResizedCrop(64),
                                                             # transforms.RandomCrop(64, padding=4),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                              normalize ]))
        val_dataset = torchvision.datasets.ImageFolder(testdir,
                                                       transforms.Compose([
                                                           # transforms.Resize(64),
                                                           # transforms.RandomResizedCrop(224),
                                                           transforms.ToTensor(),
                                                           normalize ]))
        return train_dataset, val_dataset

    elif data_config.dataset == 'imagenet':
        traindir = data_config.data_path+'/ILSVRC/train'
        valdir =data_config.data_path+'/ILSVRC/val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        jittering =aug.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = aug.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])
        train_dataset = torchvision.datasets.ImageFolder(traindir,
                                                         transforms.Compose([
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             jittering, lighting, normalize, ]))
        val_dataset = torchvision.datasets.ImageFolder(valdir,
                                                         transforms.Compose([
                                                             transforms.Resize(256),
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.ToTensor(),
                                                             normalize, ]))
        return train_dataset, val_dataset
    elif data_config.dataset == 'scoliosis':
        # traindir = data_config.data_path + '/train'
        # valdir = data_config.data_path + '/test'
        normalize = transforms.Normalize(mean=[0.64, 0.53, 0.43],
                                         std=[0.20, 0.19, 0.19])
        jittering = aug.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
        lighting = aug.Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize])
        train_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))#fa_reduced_cifar10,autoaug_paper_cifar10,fa_reduced_imagenet
        train_dataset =ScoliosisDataset(data_config.data_path,
                                        transform=train_transforms,#,jittering, lighting,transforms.RandomHorizontalFlip(),
                                        train=True)
        val_dataset = ScoliosisDataset(data_config.data_path,
                                         target_transform=test_transforms,
                                         train=False)
        return train_dataset, val_dataset
    elif data_config.dataset == 'SCUT-FBP5500':
        data_path=data_config.data_path
        trainfile = data_config.label_file + '/train.txt'
        valfile = data_config.label_file + '/test.txt'
        normalize = transforms.Normalize(mean=[0.22, 0.37, 0.73],
                                         std=[1.61, 1.75, 1.80])

        train_dataset =FacialAttractionDataset(data_path,trainfile,
                                        transform=transforms.Compose([
                                            transforms.Resize(224),
                                            # transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),normalize]),
                                        )
        val_dataset = FacialAttractionDataset(data_path,valfile ,
                                         transform=transforms.Compose([
                                             transforms.Resize(224),
                                             transforms.ToTensor(),normalize]),
                                         )
        return train_dataset, val_dataset

    elif data_config.dataset == 'sco_fa':
        source_dir=data_config.source_dir
        taget_dir=data_config.taget_dir

        trainfile = data_config.label_file + '/train.txt'
        valfile = data_config.label_file + '/test.txt'

        source_normalize = transforms.Normalize(mean=[0.22, 0.37, 0.73],
                                         std=[1.61, 1.75, 1.80])
        target_normalize = transforms.Normalize(mean=[0.64, 0.53, 0.43],
                                         std=[0.20, 0.19, 0.19])
        source_transforms=transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            source_normalize])
        target_transforms =transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            target_normalize])
        # source_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        target_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        train_dataset =ScoandFaDataset(source_dir=source_dir,
                                       taget_dir=taget_dir+'train',
                                       label_file=trainfile,
                                       source_transform=source_transforms,
                                       target_transform=target_transforms
                                        )
        val_dataset = ScoandFaDataset(source_dir=source_dir,
                                       taget_dir=taget_dir+'test',
                                       label_file=valfile,
                                       source_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),source_normalize]),
                                       target_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),target_normalize]),
                                        )
        return train_dataset, val_dataset
    elif data_config.dataset == 'scofa':
        source_dir=data_config.source_dir
        taget_dir=data_config.taget_dir

        source_normalize = transforms.Normalize(mean=[0.22, 0.37, 0.73],
                                         std=[1.61, 1.75, 1.80])
        target_normalize = transforms.Normalize(mean=[0.64, 0.53, 0.43],
                                         std=[0.20, 0.19, 0.19])
        source_transforms=transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            source_normalize])
        target_transforms =transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            target_normalize])
        # source_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        target_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        train_dataset =ScoandFaNshotDataset(source_dir=source_dir+'train',
                                       taget_dir=taget_dir+'train',
                                       source_transform=source_transforms,
                                       target_transform=target_transforms
                                        )
        val_dataset = ScoandFaNshotDataset(source_dir=source_dir+'test',
                                       taget_dir=taget_dir+'test',
                                       source_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),source_normalize]),
                                       target_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),target_normalize]),
                                        )
        return train_dataset, val_dataset
    elif data_config.dataset == 'megaage_asian':
        train_path = data_config.data_path+'train'
        val_path = data_config.data_path+'test'
        trainfile = data_config.label_file + 'train_age.txt'
        valfile = data_config.label_file + 'test_age.txt'
        normalize = transforms.Normalize(mean=[0.54, 0.47, 0.44],
                                          std=[0.29, 0.28, 0.28])

        train_dataset = MegaAsiaAgeDataset(train_path, trainfile,
                                                transform=transforms.Compose([
                                                    # transforms.Resize(224),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize]),
                                                )
        val_dataset = MegaAsiaAgeDataset(val_path, valfile,
                                              transform=transforms.Compose([
                                                  transforms.Resize(256),
                                                  transforms.RandomResizedCrop(224),
                                                  # transforms.Resize(224),
                                                  transforms.ToTensor(),
                                                  normalize]),
                                              )
        return train_dataset, val_dataset

    else:
        raise Exception('unknown dataset: {}'.format(data_config.dataset))


