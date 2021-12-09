
## StochDepth
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/stochdepth.png" width=50% />

### Usage
```python
from autoaug.stodepth import resnet18_StoDepth_lineardecay
...
model = resnet18_StoDepth_lineardecay(num_classes=100)
...
```

## label smoothing
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/image.png" width=50% />

### Usage
```python
from autoaug.label_smoothing import LabelSmoothingCrossEntropy
...
criterion = LabelSmoothingCrossEntropy()
...
```
## Cutout
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/cutout.png" width=50% />

### Usage
```python
from autoaug.cutout import Cutout

#if args.cutout:
#   train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
#dataset = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=train_transform)
#or
#dataset = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_train)
#dataset = Cutout(dataset, n_holes=args.n_holes, length=args.length)
#or
cutout=Cutout(n_holes=args.n_holes, length=args.length)
...
for _ in range(num_epoch):
    for input, target in loader:
        input=cutout(input)
...
```
## DropBlock
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/dropblock.png" width=50% />

### Usage
```python
from autoaug.dropblock.resnet18_dropblock import ResNet18
...
self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0., block_size=5),
            start_value=0.0,
            stop_value=0.25,
            nr_steps=5e3
        )
...
model = ResNet18(num_classes=100)
```
## Mixup
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/mixup.png" width=50% />

### Usage
```python
from autoaug.mixup import mixup_data,mixup_criterion
...
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
...
```

## Manifold Mixup
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/mainfold_mixup.png" width=50% />

### Usage
```python
from autoaug.resnet18_manifold_mixup import ResNet18

model = ResNet18(num_classes=100)
...
    lame = np.random.beta(1, 1)
    rand_index = torch.randperm(b)  # 打乱索引
    target_a = batch_labels
    target_b = batch_labels[rand_index]
    r = np.random.rand(1)
...
    predicted = model(inputs, rand_index, r,lame)
    loss = lame * self.myloss(predicted, target_a) + (1-lame) * self.myloss(predicted, target_b)
...
```
## ShakeDrop
<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/shakedrop.png" width=50% />

### Usage
```python
from autoaug.resnet18_shakedrop import ResNet18

model = ResNet18(num_classes=100)

```

## cutmix

<img src="https://github.com/ZhugeKongan/-DataAug-and-NetRegularization/blob/main/data/cutmix.png" width=50% />

### Usage

```python
from autoaug.cutmix import CutMix,CutMixCrossEntropyLoss
...

dataset = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_train)
dataset = CutMix(dataset, num_class=100, beta=1.0, prob=0.5, num_mix=2)    # this is paper's original setting for cifar.
...

criterion = CutMixCrossEntropyLoss(True)
for _ in range(num_epoch):
    for input, target in loader:    # input is cutmixed image's normalized tensor and target is soft-label which made by mixing 2 or more labels.
        output = model(input)
        loss = criterion(output, target)
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#else#
        lame = np.random.beta(1, 1)
        rand_index = torch.randperm(b)  # 打乱索引
        target_a = batch_labels
        target_b = batch_labels[rand_index]
        r = np.random.rand(1)

        if r < 0.0:
            bbx1, bby1, bbx2, bby2 = rand_bbox(batch_imgs.size(), lame)
            batch_imgs[:, :, bbx1:bbx2, bby1:bby2] = batch_imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
            predicted = self.model(batch_imgs, rand_index, r)
            loss =lam * self.myloss(predicted, target_a) + (1-lam) * self.myloss(predicted, target_b)
```
## Result

### ResNet18 + StochDepth|label_smoothing|Cutout|DropBlock|Mixup|Manifold Mixup|ShakeDrop|CutMix *CIFAR-100*

|  Model                  | Top-1 acc(@200epoch) | Top-5 acc |
|---------------------------------|------------:|------------|
|  ResNet18               | 77.70     | 93.89      |
| + StochDepth            | 77.85     | 94.93      | 
| + label_smoothing       | 79.11     | 94.42      | 
| + Cutout                | 78.22     | 94.41      |
| + DropBlock             | 78.12     | 94.85      |
| + Mixup                 |79.63      | 94.78      |
| + Manifold Mixup        | 80.28     | 94.96      |
| + ShakeDrop             | 78.98     | 95.00      |
| + CutMix                | 80.72     | 95.86      |



## Reference

- Official Paper
  - Deep Networks with Stochastic Depth
  - Rethinking the Inception Architecture for Computer Vision
  - Improved regularization of convolutional neural networks with cutout
  - DropBlock: A regularization method for convolutional networks
  - mixup: Beyond Empirical Risk Minimization
  - Manifold Mixup: Better Representations by Interpolating Hidden States
  - ShakeDrop Regularization for Deep Residual Learning
  - CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
  - Implementation : https://github.com/clovaai/CutMix-PyTorch


