import io
import glob
import os
import numpy as np
from shutil import move
from os.path import join
from os import listdir, rmdir


# target_folder = '/disks/disk2/lishengyan/dataset/tiny-imagenet-200/val/'
# test_folder = '/disks/disk2/lishengyan/dataset/tiny-imagenet-200/test1/'
#
# os.mkdir(test_folder)
# val_dict = {}
# with open('/disks/disk2/lishengyan/dataset/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
#     for line in f.readlines():
#         split_line = line.split('\t')
#         val_dict[split_line[0]] = split_line[1]
#
# paths = glob.glob('/disks/disk2/lishengyan/dataset/tiny-imagenet-200/val/images/*')
# for path in paths:
#     file = path.split('/')[-1]
#     folder = val_dict[file]
#     if not os.path.exists(target_folder + str(folder)):
#         os.mkdir(target_folder + str(folder))
#         os.mkdir(target_folder + str(folder) + '/images')
#     if not os.path.exists(test_folder + str(folder)):
#         os.mkdir(test_folder + str(folder))
#         os.mkdir(test_folder + str(folder) + '/images')
#
# for path in paths:
#     file = path.split('/')[-1]
#     folder = val_dict[file]
#     if len(glob.glob(target_folder + str(folder) + '/images/*')) < 25:
#         dest = target_folder + str(folder) + '/images/' + str(file)
#     else:
#         dest = test_folder + str(folder) + '/images/' + str(file)
#     move(path, dest)
#
# rmdir('./tiny-imagenet-200/val/images')

target_folder = '/disks/disk2/data/SCUT-FBP5500/test/'
source_folder = '/disks/disk2/data/SCUT-FBP5500/Images/'

# os.mkdir(test_folder)
val_dict = {}
with open('/disks/disk2/data/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt', 'r') as f:
    imgs = list(map(lambda line: line.strip().split(' '), f))

print(len(imgs))


for i in range(len(imgs)):
    img_name, label = imgs[i]
    label =float(label)
    if label<2.5:
        label=0
    elif label<3.0:
        label = 1
    elif label < 3.5:
        label = 2
    else:
        label = 3
    label = str(label)
    img_path=source_folder+img_name
    dest_path=target_folder+label+'/'+img_name
    try:
        move(img_path,dest_path)
    except:
        pass




