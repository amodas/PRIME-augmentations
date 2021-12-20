import torch
import torch.nn as nn

from .cifar.resnet import ResNet18, ResNet34, ResNet50, ResNet101

from .imagenet.basic import *

from .utils import _MODELS
from .utils import get_model


class TransformLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float)[None, :, None, None]
        std = torch.as_tensor(std, dtype=torch.float)[None, :, None, None]
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return x.sub(self.mean).div(self.std)

    
class AugModel(nn.Module):
    def __init__(self, model, aug):
        super().__init__()
        self.model = model
        self.aug = aug
        self.use_augmentation = False

    def enable_augmentation(self, val):
        self.use_augmentation = val

    def forward(self, x):
        if self.model.training:
            return self.model(self.aug(x))
        return self.model(x)


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


def deactivate_batchnorm_bias(m):
    if isinstance(m, nn.BatchNorm2d):
        m.bias.requires_grad_(False)
        m.running_mean.zero_()
        with torch.no_grad():
            m.bias.requires_grad_(False)
