# Torch-template-for-deep-learning
 Pytorch implementations of some **classical backbone CNNs, data enhancement, torch loss, attention, visualization and  some common algorithms **.

### Requirements

  · torch, torch-vision

  · torchsummary
  
  · other necessary

### usage
A training script is supplied in “train_baseline.py”, the arguments are in “args.py
<img src="https://github.com/ZhugeKongan/torch-template-for-deep-learning/blob/main/results/training.png" width=80% />
### autoaug: Data enhancement and CNNs regularization
    - StochDepth
    - label smoothing
    - Cutout
    - DropBlock
    - Mixup
    - Manifold Mixup
    - ShakeDrop
    - cutmix
### dataset_loader: Loaders for various datasets
```python
from dataloder.scoliosis_dataloder import ScoliosisDataset
from dataloder.facial_attraction_dataloder import FacialAttractionDataset
from dataloder.fa_and_sco_dataloder import ScoandFaDataset
from dataloder.scofaNshot_dataloder import ScoandFaNshotDataset
from dataloder.age_dataloder import MegaAsiaAgeDataset
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
```
### deployment: Deployment mode of pytorch model
    Two deployment modes of pytorch model are provided, one is web deployment and the other is C + + deployment

    Store the training weight file in ` flash_ Deployment ` folder

    Then modify ' server.py '  path

    Leverage ' client.Py ' call
### models: Various classical deep learning models

##### Classical network
    - **AlexNet**
    - **VGG**
    - **ResNet** 
    - **ResNext** 
    - **InceptionV1**
    - **InceptionV2 and InceptionV3**
    - **InceptionV4 and Inception-ResNet**
    - **GoogleNet**
    - **EfficienNet**
    - **MNasNet**
    - **DPN**

##### Attention network
    - **SE Attention**
    - **External Attention**
    - **Self Attention**
    - **SK Attention**
    - **CBAM Attention**
    - **BAM Attention**
    - **ECA Attention**
    - **DANet Attention**
    - **Pyramid Split Attention(PSA)**
    - **EMSA Attention**
    - **A2Attention**
    - **Non-Local Attention**
    - **CoAtNet**
    - **CoordAttention**
    - **HaloAttention**
    - **MobileViTAttention**
    - **MUSEAttention**  
    - **OutlookAttention**
    - **ParNetAttention**
    - **ParallelPolarizedSelfAttention**
    - **residual_attention**
    - **S2Attention**
    - **SpatialGroupEnhance Attention**
    - **ShuffleAttention**
    - **GFNet Attention**
    - **TripletAttention**
    - **UFOAttention**
    - **VIPAttention**

##### Lightweight network
    - **MobileNets:**
    - **MobileNetV2：**
    - **MobileNetV3：**
    - **ShuffleNet：**
    - **ShuffleNet V2:**
    - **SqueezeNet**
    - **Xception**
    - **MixNet**
    - **GhostNet**
    
##### GAN
    - **Auxiliary Classifier GAN**
    - **Adversarial Autoencoder**
    - **BEGAN**
    - **BicycleGAN**
    - **Boundary-Seeking GAN**
    - **Cluster GAN**
    - **Conditional GAN**
    - **Context-Conditional GAN**
    - **Context Encoder**
    - **Coupled GAN**
    - **CycleGAN**
    - **Deep Convolutional GAN**
    - **DiscoGAN**
    - **DRAGAN**
    - **DualGAN**
    - **Energy-Based GAN**
    - **Enhanced Super-Resolution GAN**  
    - **Least Squares GAN**
    - **Enhanced Super-Resolution GAN**
    - **GAN**
    - **InfoGAN**
    - **Pix2Pix**
    - **Relativistic GAN**
    - **Semi-Supervised GAN**
    - **StarGAN**
    - **Wasserstein GAN**
    - **Wasserstein GAN GP**
    - **Wasserstein GAN DIV**

##### ObjectDetection-network

    - **SSD:**
    - **YOLO:**
    - **YOLOv2:**
    - **YOLOv3:**
    - **FCOS:**
    - **FPN:**
    - **RetinaNet**
    - **Objects as Points:**
    - **FSAF:**
    - **CenterNet**
    - **FoveaBox**

##### Semantic Segmentation

    - **FCN**
    - **Fast-SCNN**
    - **LEDNet:**
    - **LRNNet**
    - **FisheyeMODNet:**
  
##### Instance Segmentation
    - **PolarMask** 
  
##### FaceDetectorAndRecognition
    - **FaceBoxes**
    - **LFFD**
    - **VarGFaceNet**

##### HumanPoseEstimation

    - **Stacked Hourglass Networks**
    - **Simple Baselines**
    - **LPN**
    
### pytorch_loss: loss for training
    - label-smooth
    - amsoftmax
    - focal-loss
    - dual-focal-loss 
    - triplet-loss
    - giou-loss
    - affinity-loss
    - pc_softmax_cross_entropy
    - ohem-loss(softmax based on line hard mining loss)
    - large-margin-softmax(bmvc2019)
    - lovasz-softmax-loss
    - dice-loss(both generalized soft dice loss and batch soft dice loss)

### tf_to_pytorch: TensorFlow to PyTorch Conversion
    This directory is used to convert TensorFlow weights to PyTorch. 
    It was hacked together fairly quickly, so the code is not the most 
    beautiful (just a warning!), but it does the job. I will be refactoring it soon.

### TorchCAM: Class Activation Mapping
    Simple way to leverage the class-specific activation of convolutional layers in PyTorch.
    
    - CAM
    - ScoreCAM
    - SSCAM
    - ISCAM
    - GradCAM
    - Grad-CAM++
    - Smooth Grad-CAM++
    - XGradCAM
    - LayerCAM
    
    
### Note
- **More modules may be added later**.

- **During the implementation process, I read a lot of codes and articles and referred to a lot of contents.
     Some have added copyright notices, and some don't remember the main references. If there is infringement, please contact to delete.**

- **I wrote some blogs（which are in Chinese） to introduce the models implemented in this project**：
    - [torch模板使用说明](https://mp.weixin.qq.com/s/nXiFM6Mila2bH7fopbsVOw)
    - [论文综述：注意力机制](https://zhuanlan.zhihu.com/p/388122250)
    - [论文综述：特征可视化](https://zhuanlan.zhihu.com/p/420954745)
    - [论文综述：数据增强&网络正则化](https://zhuanlan.zhihu.com/p/402511359)
    - [论文综述：轻量型网络](https://mp.weixin.qq.com/s/w9XKRzkxNfmNjUdlVuEyTQ)
    
    
- **Some of My Reference Repositories**：
    - https://github.com/xmu-xiaoma666/External-Attention-pytorch
    - https://github.com/eriklindernoren/PyTorch-GAN
    - https://www.zhihu.com/people/ZhugeKongan
    - https://github.com/ZhugeKongan/Attention-mechanism-implementation
    - https://github.com/ZhugeKongan/-DataAug-and-NetRegularization
    

## Write at the end
At present, the work organized by this project is indeed not comprehensive enough. As the amount of reading increases, we will continue to improve this project. Welcome everyone star to support. If there are incorrect statements or incorrect code implementations in the article, you are welcome to point out~

 
