
# TorchCAM: Class Activation Mapping

Simple way to leverage the class-specific activation of convolutional layers in PyTorch.

![gradcam_sample](https://github.com/frgfm/torch-cam/releases/download/v0.2.0/cam_example.png)


## Setup

Python 3.6 (or higher) and [pip](https://pip.pypa.io/en/stable/)/[conda](https://docs.conda.io/en/latest/miniconda.html) are required to install TorchCAM.

### Stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/torch-cam/) as follows:

```shell
pip install torchcam
```

or using [conda](https://anaconda.org/frgfm/torchcam):

```shell
conda install -c frgfm torchcam
```

### Developer installation

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source:

```shell
git clone https://github.com/frgfm/torch-cam.git
pip install -e torch-cam/.
```


## using your CAM
### CAM
[Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150): the original CAM paper

You can find the exhaustive list of supported CAM methods in the [documentation](https://frgfm.github.io/torch-cam/cams.html), then use it as follows:

```python
from torchvision.models import resnet18
from torchcam.cams import CAM
model = resnet18(pretrained=True).eval()
cam = CAM(model, 'layer4', 'fc')
with torch.no_grad(): out = model(input_tensor)
cam(class_idx=100)
```

*Please note that by default, the layer at which the CAM is retrieved is set to the last non-reduced convolutional layer. If you wish to investigate a specific layer, use the `target_layer` argument in the constructor.*

### ScoreCAM

paper:[Score-CAM:Score-Weighted Visual Explanations for Convolutional Neural Networks](https://arxiv.org/pdf/1910.01279.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import ScoreCAM
model = resnet18(pretrained=True).eval()
cam = ScoreCAM(model, 'layer4', 'fc')
with torch.no_grad(): out = model(input_tensor)
cam(class_idx=100)
```

### SSCAM
paper:[SS-CAM: Smoothed Score-CAM for Sharper Visual Feature Localization](https://arxiv.org/pdf/2006.14255.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import SSCAM
model = resnet18(pretrained=True).eval()
cam = SSCAM(model, 'layer4', 'fc')
with torch.no_grad(): out = model(input_tensor)
cam(class_idx=100)
```

### ISCAM
paper:[IS-CAM: Integrated Score-CAM for axiomatic-based explanations](https://arxiv.org/pdf/2010.03023.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import ISCAM
model = resnet18(pretrained=True).eval()
cam = ISCAM(model, 'layer4', 'fc')
with torch.no_grad(): out = model(input_tensor)
cam(class_idx=100)
```

### GradCAM
paper:[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import GradCAM
model = resnet18(pretrained=True).eval()
cam = GradCAM(model, 'layer4')
scores = model(input_tensor)
cam(class_idx=100, scores=scores)
```

### Grad-CAM++
paper:[Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/pdf/1710.11063.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import  GradCAMpp
model = resnet18(pretrained=True).eval()
cam =  GradCAMpp(model, 'layer4')
scores = model(input_tensor)
cam(class_idx=100, scores=scores)
```

### Smooth Grad-CAM++
paper:[Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models](https://arxiv.org/pdf/1908.01224.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import SmoothGradCAMpp
model = resnet18(pretrained=True).eval()
cam = SmoothGradCAMpp(model, 'layer4')
scores = model(input_tensor)
cam(class_idx=100, scores=scores)
```

### XGradCAM
paper:[Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs](https://arxiv.org/pdf/2008.02312.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import XGradCAM
model = resnet18(pretrained=True).eval()
cam = XGradCAM(model, 'layer4')
scores = model(input_tensor)
cam(class_idx=100, scores=scores)
```

### LayerCAM
paper:[LayerCAM: Exploring Hierarchical Class Activation Maps for Localization](http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf)

```python
from torchvision.models import resnet18
from torchcam.cams import LayerCAM
model = resnet18(pretrained=True).eval()
cam = LayerCAM(model, 'layer4')
scores = model(input_tensor)
cam(class_idx=100, scores=scores)
```



### Retrieving the class activation map

Once your CAM extractor is set, you only need to use your model to infer on your data as usual. If any additional information is required, the extractor will get it for you automatically.

```python
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.cams import SmoothGradCAMpp

model = resnet18(pretrained=True).eval()
cam_extractor = SmoothGradCAMpp(model)
# Get your input
img = read_image("path/to/your/image.png")
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
```

If you want to visualize your heatmap, you only need to cast the CAM to a numpy ndarray:

```python
import matplotlib.pyplot as plt
# Visualize the raw CAM
plt.imshow(activation_map.numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
```

![raw_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/raw_heatmap.png)

Or if you wish to overlay it on your input image:

```python
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
```

![overlayed_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/overlayed_heatmap.png)




## CAM Zoo

This project is developed and maintained by the repo owner, but the implementation was based on the following research papers:

- [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150): the original CAM paper
- [Grad-CAM](https://arxiv.org/abs/1610.02391): GradCAM paper, generalizing CAM to models without global average pooling. 
- [Grad-CAM++](https://arxiv.org/abs/1710.11063): improvement of GradCAM++ for more accurate pixel-level contribution to the activation.
- [Smooth Grad-CAM++](https://arxiv.org/abs/1908.01224): SmoothGrad mechanism coupled with GradCAM.
- [Score-CAM](https://arxiv.org/abs/1910.01279): score-weighting of class activation for better interpretability.
- [SS-CAM](https://arxiv.org/abs/2006.14255): SmoothGrad mechanism coupled with Score-CAM.
- [IS-CAM](https://arxiv.org/abs/2010.03023): integration-based variant of Score-CAM.
- [XGrad-CAM](https://arxiv.org/abs/2008.02312): improved version of Grad-CAM in terms of sensitivity and conservation.
- [Layer-CAM](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf): Grad-CAM alternative leveraging pixel-wise contribution of the gradient to the activation.



## reference

```bibtex
@misc{torcham2020,
    title={TorchCAM: class activation explorer},
    author={François-Guillaume Fernandez},
    year={2020},
    month={March},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/frgfm/torch-cam}}
}
```

