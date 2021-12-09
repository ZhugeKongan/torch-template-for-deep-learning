""" 
metrics utilized for evaluating multi-label classification system
originally written by Gancen and available at #TODO add the path
"""
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score, \
    classification_report, hamming_loss, accuracy_score, coverage_error, label_ranking_loss,\
    label_ranking_average_precision_score, classification_report


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""  # [128, 10],128
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # [128, 5],indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 5,128

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def class_accuracy(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""  # [128, 10],128
    maxk = topk
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # [128, 1],indices
    pred = pred.t().squeeze(0)
    # print(pred)
    # print(target)
    res = []
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in range(2):
        indices=torch.where(target==k)
        # print(indices)
        correct = torch.where(pred[indices]==k)
        # print(len(correct[0]),len(indices[0]))
        try:
            res.append(len(correct[0])*100.0 / len(indices[0]))
        except:
            res.append(0)
    return res
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        sample_prec = precision_score(true_labels, predict_labels, average='samples')
        micro_prec = precision_score(true_labels, predict_labels, average='micro')
        macro_prec = precision_score(true_labels, predict_labels, average='macro')

        return macro_prec, micro_prec, sample_prec    


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        sample_rec = recall_score(true_labels, predict_labels, average='samples')
        micro_rec = recall_score(true_labels, predict_labels, average='micro')
        macro_rec = recall_score(true_labels, predict_labels, average='macro')

        return macro_rec, micro_rec, sample_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f1 = f1_score(true_labels, predict_labels, average="macro")
        micro_f1 = f1_score(true_labels, predict_labels, average="micro")
        sample_f1 = f1_score(true_labels, predict_labels, average="samples")

        return macro_f1, micro_f1, sample_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="macro")
        micro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="micro")
        sample_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="samples")

        return macro_f2, micro_f2, sample_f2

class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        return hamming_loss(true_labels, predict_labels)

class Subset_accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        return accuracy_score(true_labels, predict_labels)


class One_error(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        row_inds = np.arange(predict_probs.shape[0])
        col_inds = np.argmax(predict_probs, axis=1)
        return np.mean((true_labels[tuple(row_inds), tuple(col_inds)] == 0).astype(int))

class Coverage_error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return coverage_error(true_labels, predict_probs)

class Ranking_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_loss(true_labels, predict_probs)

class LabelAvgPrec_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_average_precision_score(true_labels, predict_probs)

class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report
# class Mean_average_precision(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, predict_labels, true_labels):


class Accuracy_score(nn.Module):

    def __init__(self):
        super().__init__()
    def Mean_average_precision(self,preds,targets):
        # print(preds.shape,targets.shape)#(128, 17)
        map=[]
        for i in range(len(targets[0])):#17
            pred, target=preds[:,i],targets[:,i]
            z = np.argsort(-pred)
            target = target[z]
            prec = []
            rec = []
            for i in range(len(target)):
                p = float(np.sum(target[0:i + 1])) / (i + 1)
                r = float(np.sum(target[0:i + 1])) / (np.sum(target))
                # print(p, r)
                prec.append(p)
                rec.append(r)
            prec,rec=np.array(prec),np.array(rec)
            n = int(np.sum(target))
            ap=0
            if n != 0:
                for i in range(n):
                    t = float(i ) / n
                    # print(t)
                    z = np.where(rec > t)
                    # print("z",z)
                    p = np.max(prec[z])
                    ap += p / n
                map.append(ap)
        map=np.array(map)
        # print(map.mean())
        return map.mean()


    def forward(self, predict_labels, true_labels):
        # sample accuracy
        # print(predict_labels.shape, true_labels.shape)#(128, 17) (128, 17)
        LRAP = label_ranking_average_precision_score(true_labels, predict_labels)
        map=self.Mean_average_precision( predict_labels,true_labels)

        predict_labels=np.round(predict_labels)
        TP = (np.logical_and((predict_labels == 1), (true_labels == 1))).astype(int)
        FP = (np.logical_and((predict_labels == 1), (true_labels == 0))).astype(int)
        TN = (np.logical_and((predict_labels == 0), (true_labels == 0))).astype(int)
        FN = (np.logical_and((predict_labels == 0), (true_labels == 1))).astype(int)
        union = (np.logical_or((predict_labels == 1), (true_labels == 1))).astype(int)
        #基于样本的准确率
        TP_sample = TP.sum(axis=1)
        union_sample = union.sum(axis=1)
        sample_Acc = TP_sample / union_sample

        assert np.isfinite(sample_Acc).all(), 'Nan found in sample accuracy'
        TP_cls = TP.sum(axis=0)
        FP_cls = FP.sum(axis=0)
        TN_cls = TN.sum(axis=0)
        FN_cls = FN.sum(axis=0)
        assert (TP_cls + FP_cls + TN_cls + FN_cls == predict_labels.shape[0]).all(), 'wrong'

        # P R F-SCORE
        # prec = precision_score(true_labels, predict_labels, average='macro')
        # rec = recall_score(true_labels, predict_labels, average='macro')
        # f1=f1_score(true_labels, predict_labels, average="macro")
        # print(prec,rec,f1)#0.5528033300381151 0.49097155113489055 0.47989612765349365

        prec =(TP_cls  / (TP_cls + FP_cls ))
        prec = prec[np.where((TP_cls + FP_cls ) != 0)].mean()
        rec = (TP_cls / (TP_cls + FN_cls))
        rec = rec[np.where((TP_cls + FN_cls) != 0)].mean()
        f1=(2*prec*rec)/(prec+rec)
        # print(prec, rec, f1)#0.7831380508873297 0.7587742153902854 0.7707636460261783

        macro_Acc = (TP_cls + TN_cls) / (TP_cls + FP_cls + TN_cls + FN_cls)
        # micro_Acc = (TP_cls.mean() + TN_cls.mean()) / (TP_cls.mean() + FP_cls.mean() + TN_cls.mean() + FN_cls.mean())

        # print(LRAP,map,sample_Acc.mean(),macro_Acc.mean(),prec, rec, f1)
        return LRAP,map,sample_Acc.mean(),macro_Acc,prec, rec, f1

if __name__ == "__main__":
    acc = Accuracy_score()

    aa = (np.random.randn(100,20)>=0).astype(int)

    bb = (np.random.randn(100,20)>=0).astype(int)

    samp_acc, macro_acc, micro_acc = acc(aa, bb)

    print(samp_acc)
    print(macro_acc)
    print(micro_acc)











