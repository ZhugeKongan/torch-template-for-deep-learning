# -*- coding:UTF-8 -*-
"""
training classifying task with CNNs

"""

import os
import warnings
import functools
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.optim as optim
from torchsummary import summary
from torch import sigmoid,softmax
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

from args import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.algorithm_utils import *
from dataloder import load_dataset
from metrics import Accuracy_score, AverageMeter,accuracy

from models.ClassicNetwork.ResNet import ResNet18

'''***********- Hyper Arguments-*************'''
warnings.filterwarnings("ignore")
# device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# device=torch.device(device_name)
if data_config.rand_seed>0:
    init_rand_seed(data_config.rand_seed)

print("***********- ***********- READ DATA and processing-*************")
train_dataset,val_dataset = load_dataset(data_config)
x,y= train_dataset[0]
print('input:',x.size(),'lable:',y)#[3, 32, 32]) 6

print("***********- loading model -*************")
if(len(data_config.gpus)==0):#cpu
    model = ResNet18(dataset=data_config.dataset, num_classes=data_config.num_class)
elif(len(data_config.gpus)==1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_config.gpus[0])
    model = ResNet18(dataset=data_config.dataset, num_classes=data_config.num_class).cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in data_config.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model =ResNet18(dataset=data_config.dataset,num_classes=data_config.num_class).cuda()
    gpus = [i for i in range(len(data_config.gpus))]
    model = torch.nn.DataParallel(model, device_ids=gpus)
model_path=data_config.MODEL_PATH+'/{}_best_params.pkl'.format(data_config.model_name)
# model.load_state_dict(torch.load(model_path))

optimizer = eval(data_config.optimizer)(model.parameters(),**data_config.optimizer_parm)
scheduler = eval(data_config.scheduler)(optimizer,**data_config.scheduler_parm)
loss_f=eval(data_config.loss_f)()
loss_dv=eval(data_config.loss_dv)()
loss_fn = eval(data_config.loss_fn)()

# summary(net, (3, 224, 224))
'''***********- VISUALIZE -*************'''
# #tensorboard --logdir=<your_log_dir>
writer = SummaryWriter('runs/'+data_config.model_name)
# # get some random training images
# images, labels = next(iter(train_dataset))cd
# images=torch.unsqueeze(images.permute(2,0,1),0).cuda()
# writer.add_graph(model, images)
# writer.close()

'''***********- trainer -*************'''
class trainer:
    def __init__(self, loss_f,loss_dv,loss_fn, model, optimizer, scheduler, config):
        self.loss_f = loss_f
        self.loss_dv = loss_dv
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def batch_train(self, batch_imgs, batch_labels, epoch):
        predicted = self.model(batch_imgs)
        loss =self.myloss(predicted, batch_labels)
        predicted = softmax(predicted, dim=-1)
        del batch_imgs, batch_labels
        return loss, predicted

    def train_epoch(self, loader,warmup_scheduler,epoch):
        self.model.train()
        tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("\n************Training*************")
        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            # print("data",imgs.size(), labels.size())#[128, 3, 32, 32]) torch.Size([128]
            if (len(data_config.gpus) > 0):
                imgs, labels=imgs.cuda(), labels.cuda()
            # print(self.optimizer.param_groups[0]['lr'])
            loss, predicted = self.batch_train(imgs, labels, epoch)
            losses.update(loss.item(), imgs.size(0))
            # print(predicted.size(),labels.size())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            err1, err5 = accuracy(predicted.data, labels, topk=(1, 5))
            top1.update(err1.item(), imgs.size(0))
            top5.update(err5.item(), imgs.size(0))

            tqdm_loader.set_description('Training: loss:{:.4}/{:.4} lr:{:.4} err1:{:.4} err5:{:.4}'.
                                        format(loss, losses.avg, self.optimizer.param_groups[0]['lr'],top1.avg, top5.avg))
            if epoch <= data_config.warm:
                warmup_scheduler.step()
            # if batch_idx%1==0:
            #     break
        return top1.avg, top5.avg, losses.avg

    def valid_epoch(self, loader, epoch):
        self.model.eval()
        # tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("\n************Evaluation*************")
        for batch_idx, (batch_imgs, batch_labels) in enumerate(loader):
            with torch.no_grad():
                if (len(data_config.gpus) > 0):
                    batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
                predicted= self.model(batch_imgs)
                loss = self.myloss(predicted, batch_labels).detach().cpu().numpy()
                predicted = softmax(predicted, dim=-1)
                losses.update(loss.item(), batch_imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), batch_imgs.size(0))
                top5.update(err5.item(), batch_imgs.size(0))

        return top1.avg, top5.avg, losses.avg

    def adjust_learning_rate(self,optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr=data_config.lr
        if data_config.dataset.startswith('cifar'):
            # lr = data_config.lr * (0.1 ** (epoch // (data_config.epochs * 0.3))) * (0.1 ** (epoch // (data_config.epochs * 0.75)))
            if epoch < 60:
                lr = data_config.lr
            elif epoch < 120:
                lr = data_config.lr * 0.2
            elif epoch < 160:
                lr = data_config.lr * 0.04
            else:
                lr = data_config.lr * 0.008
        elif data_config.dataset == ('imagenet'):
            if data_config.epochs == 300:
                lr = data_config.lr * (0.1 ** (epoch // 75))
            else:
                lr = data_config.lr * (0.1 ** (epoch // 30))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def myloss(self,predicted,labels):
        # print(predicted.size(),labels.size())#[128, 10]) torch.Size([128])
        loss = self.loss_f(predicted,labels)
        return loss

    def run(self, train_loder, val_loder,model_path):
        best_err1, best_err5 = 100,100
        start_epoch=0
        top_score = np.ones([5, 3], dtype=float)*100
        top_score5 = np.ones(5, dtype=float) * 100
        iter_per_epoch = len(train_loder)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * data_config.warm)
        # model, optimizer, start_epoch=load_checkpoint(self.model,self.optimizer,model_path)
        for e in range(self.config.epochs):
            e=e+start_epoch+1
            print("------model:{}----Epoch: {}--------".format(self.config.model_name,e))
            if e > data_config.warm:
                self.scheduler.step(e)
                # adjust_learning_rate(self.optimizer,e,data_config.model_type)
            # torch.cuda.empty_cache()
            _, _, train_loss = self.train_epoch(train_loder,warmup_scheduler,e)
            err1, err5, val_loss=self.valid_epoch(val_loder,e)
            #
            print("\nval_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(val_loss, err1, err5))

            if err1 <= best_err1:
                best_err1 = err1
                print('Current Best (top-1 error):',best_err1)
            if err5 <= best_err5:
                best_err5 = err5
                print('Current Best (top-5 error):', best_err5)

            if err1 < top_score[4][2]:
                top_score[4]=[e,val_loss,err1]
                z = np.argsort(top_score[:, 2])
                top_score = top_score[z]
                best_err1 = save_checkpoint(self.model, self.optimizer, e, val_loss=err1, check_loss=best_err1,
                                            savepath=self.config.MODEL_PATH, m_name=self.config.model_name)
            if err5 < top_score5[4]:
                top_score5[4]=err5
                z = np.argsort(top_score5)
                top_score5 = top_score5[z]
                # print(top_score5)
            if(data_config.tensorboard):
                writer.add_scalar('training loss', train_loss, e)
                writer.add_scalar('valing loss', val_loss, e)
                writer.add_scalar('err1', err1, e)
                writer.add_scalar('err5', err5, e)

        writer.close()
        print('\nbest score:{}'.format(data_config.model_name))
        for i in range(5):
            print(top_score[i])
        print(top_score5,top_score[:, 0])
        print('Best(top-1 and 5 error):',top_score[:, 1].mean(), best_err1, best_err5)

        print("best accuracy:\n avg_acc1:{:.4f} | best_acc1:{:.4f} | avg_acc5:{:.4f} | | best_acc5:{:.4f} ".
              format(100 - top_score[:, 2].mean(), 100 - best_err1, 100 - top_score5.mean(), 100 - best_err5))

# print('''***********- training -*************''')
Trainer = trainer(loss_f,loss_dv,loss_fn,model,optimizer,scheduler,config=data_config)
train = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)
val = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.WORKERS, pin_memory=True)
Trainer.run(train,val,model_path)
