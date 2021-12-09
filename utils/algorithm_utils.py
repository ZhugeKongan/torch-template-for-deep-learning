# -*- coding:UTF-8 -*-
"""
utils for algorithm

"""

import os
import cv2
import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import label_ranking_average_precision_score

'''***********-正则化、标准化-*************'''
def img_norm(img):
    min, max =np.min(img), np.max(img)
    img = img - min
    img = img / (max-min)
    # img= (img * 255.0).astype('uint8')
    return img
def img_stan(img,mean, std):
    # mean, std =np.mean(img), np.std(img)
    img = img - mean
    img = img / std
    # img= (img * 255.0).astype('uint8')
    return img
def img_Normalize(img,mean,std):
    img = img - mean
    img = img / std
    return img
'''***********-聚类-*************'''
def clustering(img, num=5, size=120):
    # 图像二维像素转换为一维
    # plt.imshow(img)
    # plt.show()
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means聚类 聚集成2类
    compactness, labels2, centers2 = cv2.kmeans(data, num, None, criteria, 10, flags)
    # print(compactness)
    label = labels2.reshape(size, size)
    # plt.imshow(label)
    # plt.show()
    output = np.zeros((num+1, size, size, 3))
    output[num]=img
    for i in range(0, size):
        for j in range(0, size):
            t = label[i][j]
            output[t][i][j] = img[i][j]

    # for i in range(0, num):
    #     plt.imshow(output[i])
    #     plt.show()
    #     get_proportion(output[i])
    # print(np.shape(output))
    return output
def fuz_clustering(img, h=120, w=120, dim=3, ncenters=5):
    # 将数据转化为一维
    # print(np.shape(img))
    data = img.reshape((dim, -1))
    print(data)
    alldata = np.float32(data)
    # print(np.shape(data))
    # FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # (ncenters，-1)->(k,h,w,dim)
    # print(np.shape(u))
    u = u.reshape((ncenters, h, w))
    output = np.ones((ncenters, h, w, dim))

    for i in range(ncenters):
        for j in range(h):
            for k in range(w):
                output[i][j][k] = u[i][j][k] * img[j][k]
                # print(output[i][j][k], u[i][j][k], img[j][k])
    # print(np.shape(output))
    return cntr,output

'''***********-精确率与召回率-*************'''
def prec_rec(pred,target):
    # print(pred,target)
    prec=[]
    rec=[]
    z=np.argsort(-pred)
    # print(z)
    target=target[z]
    # print(target)
    # if target[0]==0:
    #     # print(len(target))
    #     target[:len(target)-1]=target[1:]
    #     target[len(target)-1]=0

    # pred=sorted(pred,reverse=True)
    # print("s",pred)
    for i in range(len(target)):
        p=float(np.sum(target[0:i+1]))/(i+1)
        r=float(np.sum(target[0:i+1]))/(np.sum(target))
        # print(p, r)
        prec.append(p)
        rec.append(r)
    # print(prec,rec)
    return np.array(prec),np.array(rec)

'''***********-MAP评价指标-*************'''
def voc_map(prec,rec,n,apflag):
    ap=0
    if apflag==True:
        # 11 point metric
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            # print(t,p)
            ap +=  p / 11.

    else:
        for i in np.arange(n):
            t = (i + 1) / n
            # print(t)
            z = np.where(rec >= t)
            # print("z",z)
            p = np.max(prec[z])
            ap += p/n
    # else:
    #     mrec = np.concatenate(([0.], rec, [1.]))
    #     mpre = np.concatenate(([0.], prec, [0.]))
    #     for i in range(mpre.size - 1, 0, -1):
    #         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    #     i = np.where(mrec[1:] != mrec[:-1])[0]
    #     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # print(ap/n)
    return ap
# def f_score(preds,targets):
#     print(preds.shape)#(128*100, 43)
#     for i in range(targets.shape[1]):
#         pred, target=preds[:,i],targets[:,i]
#         z = np.argsort(-pred)
#         # print(z)
#         target = target[z]
#         L = np.sum(target)
#         tp = np.sum(target[:L])
#         tn = len(target) - L -np.sum(target[L:])
#         fp=np.sum(target[L:])
#         fn=L-np.sum(target[:L])
#         print(len(target),L,tp,tn,fp,fn)
#         p=tp/(tp+fp)
#         r=tp/(tp+fn)
#         f1=2*p*r/(p+r)
#         print(p,r,f1)

def batch_map(pred,target,apflag=False):
    # print(pred.shape)#(128, 43)
    ap=0
    m=0
    mean_prec= 0
    t1=t5=0
    LRAP= label_ranking_average_precision_score(target,pred)
    for i in range(pred.shape[0]):
        prec, rec = prec_rec(pred[i],target[i])
        t1+=prec[0]
        t5+=prec[4]
        # print(prec, rec)
        n = np.sum(target[i])
        # print(int(n - 1),prec[int(n - 1)],rec[int(n - 1)])
        if n!=0:
            mean_prec = (mean_prec * i + prec[int(n - 1)]) / (i + 1)
            # mean_rec = (mean_rec * i + rec[int(n - 1)]) / (i + 1)
            # f_score = (2 * mean_prec * mean_rec) / (mean_prec + mean_rec)
            m+=1
            ap += voc_map(prec, rec, n,apflag)
            # print(ap)
    # print('F_score:',mean_prec,mean_rec,f_score)
    return ap/m,mean_prec,LRAP,t1/pred.shape[0],t5/pred.shape[0]


from sklearn.metrics import f1_score, precision_score, recall_score #fbeta_score, \
    # classification_report, hamming_loss, accuracy_score, coverage_error, label_ranking_loss,\
    # label_ranking_average_precision_score, classification_report

def f_score(true_labels, predict_labels):
    predict_labels=np.round(predict_labels)
    sample_prec = precision_score(true_labels, predict_labels, average='samples')
    micro_prec = precision_score(true_labels, predict_labels, average='micro')
    macro_prec = precision_score(true_labels, predict_labels, average='macro')
    # print(sample_prec,micro_prec,macro_prec)
    sample_rec = recall_score(true_labels, predict_labels, average='samples')
    micro_rec = recall_score(true_labels, predict_labels, average='micro')
    macro_rec = recall_score(true_labels, predict_labels, average='macro')
    # print(sample_rec,micro_rec,macro_rec)
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    micro_f1 = f1_score(true_labels, predict_labels, average="micro")
    sample_f1 = f1_score(true_labels, predict_labels, average="samples")
    # print(sample_f1,micro_f1,macro_f1)
    return (micro_prec,micro_rec,micro_f1),(macro_prec,macro_rec,macro_f1)
# y_true = np.array([[0, 1, 0, 1, 1, 0],[0, 1, 0, 0, 1, 0]])
# y_pred = np.array([[0, 1, 1, 1, 0, 1],[0, 0, 1, 0, 1, 1]])#.transpose(1,0)
# # print(y_true.shape)
# a,b=f_score(y_true,y_pred)
# print(a,b)
# print(precision_score(y_true, y_pred, average='macro'))#3/6
# print(precision_score(y_true, y_pred, average='micro'))#3/7
# print(precision_score(y_true, y_pred, average='samples'))#(2/4+1/3)/2
# a=[[0.65,0.11,0.76,0.55,0.94,0.28,0.93,0,0.84,0.42],[0.13,0.71,0.26,0.55,0.44,0.81,0.13,0.58,0.14,0.92],
#    [0.65,0.11,0.76,0.55,0.94,0.28,0.93,0.15,0.84,0.42],[0.13,0.71,0.26,0.55,0.44,0.81,0.13,0.58,0.14,0.92],
#    [0.65,0.11,0.76,0.55,0.94,0.28,0.93,0.15,0.84,0.42],[0.13,0.71,0.26,0.55,0.44,0.81,0.13,0.58,0.14,0.92],
#    [0.65,0.11,0.76,0.55,0.94,0.28,0.93,0.15,0.84,0.42],[0.13,0.71,0.26,0.55,0.44,0.81,0.13,0.58,0.14,0.92],
#    [0.65,0.11,0.76,0.55,0.94,0.28,0.93,0.15,0.84,0.42],[0.13,0.71,0.26,0.55,0.44,0.81,0.13,0.58,0.14,0.92],
#    [0.15,0.71,0.76,0.55,0.94,0.28,0.93,0.15,0.84,0.42],[0.93,0.11,0.26,0.55,0.44,0.81,0.13,0.58,0.14,0.92]
#    ]
# b=[[1,0,1,0,0,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1],
#    [1,0,1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1],
#    [1,0,1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1],
#    [1,0,1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1],
#    [1,0,1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1],
#    [1,0,1,0,0,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1]
#    ]
# # import torch
# # import torch.nn as nn
# a,b=np.array(a),np.array(b) #.transpose(1,0)torch.from_numpy(
# f=f_score(b,a)
# # a.astype(np.int)
# print("a",a)
# a= np.round(a)
# print("a",a)
# p=precision_score(b, a, average="micro")#samples,macro,micro
# r=recall_score(b, a, average="samples")
# f1=f1_score(b, a, average="samples")
# print(p,r,f1)
# class Precision_score(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, predict_labels, true_labels):
#
#         sample_prec = precision_score(true_labels, predict_labels, average='samples')
#         micro_prec = precision_score(true_labels, predict_labels, average='micro')
#         macro_prec = precision_score(true_labels, predict_labels, average='macro')
#
#         return macro_prec, micro_prec, sample_prec
#
#
# class Recall_score(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, predict_labels, true_labels):
#
#         sample_rec = recall_score(true_labels, predict_labels, average='samples')
#         micro_rec = recall_score(true_labels, predict_labels, average='micro')
#         macro_rec = recall_score(true_labels, predict_labels, average='macro')
#
#         return macro_rec, micro_rec, sample_rec
#
#
# class F1_score(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, predict_labels, true_labels):
#
#         macro_f1 = f1_score(true_labels, predict_labels, average="macro")
#         micro_f1 = f1_score(true_labels, predict_labels, average="micro")
#         sample_f1 = f1_score(true_labels, predict_labels, average="samples")
#
#         return macro_f1, micro_f1, sample_f1
#
# p=f1_score(b, a, average="macro")
# r=Recall_score(b,a)
# f1=F1_score(b,a)
# print(p,r,f1)
# # c,d=np.array(a).transpose(1,0),np.array(b).transpose(1,0)
# nap=batch_map(a,b)
# # ap=batch_map(c,d,True)
# print(nap)
# # from sklearn.metrics import label_ranking_average_precision_score
# LRAP1 = label_ranking_average_precision_score(b, a)
# LRAP2 = label_ranking_average_precision_score(d, c)
# print(LRAP1,LRAP2)