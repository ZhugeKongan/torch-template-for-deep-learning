# -*- coding:UTF-8 -*-
"""
Fetching arguments in args.yaml

"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def fetch_args(yaml_path ):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        args = f.read()
        args = yaml.load(args)
        return args

def init_rand_seed(rand_seed):
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_folder(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def read_csv(path):
    return pd.read_csv(path)
def split_train_test_csv(path):
    df = read_csv(path)
    # print(len(df))  # 2100
    train_df = df.copy()
    test_df = []
    for index, row in tqdm(train_df.iterrows()):
        img_name = row.loc['IMAGE\LABEL']
        # print(img_name)
        a = int(img_name[-2:])
        # print(a)
        if a<19:
            test_df.append(row)
            train_df = train_df.drop(index=[index])
    test_df = pd.DataFrame(test_df, columns=df.columns)
    # train_df.to_csv('/disks/disk2/lishengyan/MyProject/SIRS_Multi_Labels/dataset/train_df.csv', index=False)
    # test_df.to_csv('/disks/disk2/lishengyan/MyProject/SIRS_Multi_Labels/dataset/test.csv', index=False)
    return train_df,test_df
def comput_loss_w():
    # df = read_csv(path)
    # data = df.values[:, 1:]  # .astype(np.int)
    # print(data[1])
    # print(data.shape)
    # s = np.sum(data, axis=0,dtype=float)
    # print(list(s))
    s=np.array([72,1107,2027,94])
    step1 = np.log2(np.max(s) / s)
    print(step1)
    for i in range(len(step1)):
        if step1[i]<0.5:
            step1[i]=0.5
        elif step1[i]<1:
            step1[i] = 1
    print(step1)
    return list(step1)

# comput_loss_w()
'''***********-  Label Smoothing-*************'''
# def smooth_one_hot(true_labels, classes, smoothing=0.0):
#     """
#     if smoothing == 0, it's one-hot method
#     if 0 < smoothing < 1, it's smooth method
#
#     """
#     confidence = 1.0 - smoothing
#     tmp=min(classes)
#     with torch.no_grad():
#         true_dist = torch.empty(size=true_labels.size(), device=true_labels.device)
#         for i in range(true_labels.size(1)):
#             true_dist[:,i]=tmp*smoothing / classes[i]
#
#         # print(true_dist)
#         # true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
#     return true_dist+true_labels*confidence

# predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                                  [0, 0.9, 0.2, 0.2, 1],
#                                  [1, 0.2, 0.7, 0.9, 1]])
# label=torch.LongTensor([[0, 1, 1, 1, 0],
#                        [0, 1, 0, 0, 1],
#                        [1, 0, 1, 1, 1]])
# data_num=[100, 754, 713, 897, 116, 105, 100, 103, 977, 102, 1331, 291, 101, 103, 100, 1021, 208]
# smooth_label=smooth_one_hot(label,5,0.1)
# print(smooth_label)

# train_file='/disks/disk2/lishengyan/MyProject/SIRS_Multi_Labels/dataset/multi-labels.csv'
# comput_loss_w(train_file)

def merge_csv(df1,df2):
    for index, row in tqdm(df1.iterrows()):
        f = row.id
        fnames = f.split('.jpg')[0].split('/')[-3:]
        # print(fnames)
        right = df2[df2['SOPInstanceUID'] == fnames[2]]
        # print( right)
        right['path'] = f
        # right.loc[index, 'path'] = f
        if index == 0:
            mini_train = right
        else:
            mini_train = mini_train.append(right, ignore_index=True)
        # print(min_train.head())

        # order_train=order_train.dropna(subset=['label'], how='any')
        # order_train=order_train.drop_duplicates(subset=['label'], keep='first')
        # # order_train=order_train[~order_train['label'].isin(['null'])]
        # order_train = order_train.drop(columns=['Unnamed: 0', ], axis=1).rename(
        #     columns={'SOPInstanceUID_y': 'SOPInstanceUID'})

        for index, row in tqdm(df1.iterrows()):
            left=df1[df1['SOPInstanceUID']==row['SOPInstanceUID']]
            right=df2[df2['SOPInstanceUID']==row['SOPInstanceUID']]
            # print(left, right)
            result = pd.merge(left, right, on='SOPInstanceUID')
            # result = pd.concat([order_train, df], axis=1,
            #                    join_axes=[f.split('.jpg')[0].split('/')[-1] for f in order_train.label.values.tolist()])
            # print(result)
            if index==0:
                order_train = result
            else:
                order_train=order_train.append(result, ignore_index=True)
        try:
            order_train.append(result,ignore_index=True)
        except Exception as e:
            order_train = result
        print(order_train)

        mini_train = mini_train.append(mini_train, ignore_index=True).sample(frac=1).reset_index(drop=True)
        mini_train.to_csv('mini_train.csv', index=False)

