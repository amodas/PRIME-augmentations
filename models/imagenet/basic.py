import torchvision.models as tmodels

from .. import utils


utils.register_model(
    cls=tmodels.resnet50, 
    dataset='imagenet', name='resnet50'
)

utils.register_model(
    cls=tmodels.resnet18, 
    dataset='imagenet100', name='resnet18'
)

utils.register_model(
    cls=tmodels.resnet50, 
    dataset='imagenet100', name='resnet50'
)

utils.register_model(
    cls=tmodels.wide_resnet50_2, 
    dataset='imagenet100', name='wide_resnet50_2'
)

utils.register_model(
    cls=tmodels.resnext50_32x4d, 
    dataset='imagenet100', name='resnext50'
)
