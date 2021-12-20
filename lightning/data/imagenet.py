import os
import tempfile
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
from robustness.datasets import CustomImageNet
from torch.utils.data import ConcatDataset, DataLoader

from . import utils


class ImageNet(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000)],
            **kwargs,
        )


class ImageNetC(CustomImageNet):
    def __init__(self, data_path, corruption_type, severity, **kwargs):
        tmp_data_path = tempfile.mkdtemp()
        os.symlink(
            os.path.join(data_path, corruption_type, str(severity)),
            os.path.join(tmp_data_path, "test"),
        )
        super().__init__(
            data_path=tmp_data_path,
            custom_grouping=[[label] for label in range(0, 1000)],
            **kwargs,
        )


class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)] if '100' not in data_path else 
                            [[label] for label in range(0, 100)],
            **kwargs,
        )


class ImageNet100C(CustomImageNet):
    def __init__(self, data_path, corruption_type, severity, **kwargs):
        tmp_data_path = tempfile.mkdtemp()
        os.symlink(
            os.path.join(data_path, corruption_type, str(severity)),
            os.path.join(tmp_data_path, "test"),
        )
        super().__init__(
            data_path=tmp_data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)] if '100' not in data_path else 
                            [[label] for label in range(0, 100)],
            **kwargs,
        )
        

@utils.register_dataset(name='imagenet')
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = './', train_batch_size=256, 
        test_batch_size=512, num_workers=4, pin_memory=True, 
        shuffle_train=True,
    ):
        
        super().__init__()
        self.data_dir = data_dir
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.num_classes = 1000
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = T.Compose([
            T.ToTensor(), T.Resize(256), T.CenterCrop(224), 
            T.Normalize(self.mean, self.std)])
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dims = (3, 224, 224)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train

    def setup(self, stage: Optional[str] = None):
        self.dataset = ImageNet(self.data_dir)
        train_loader, test_loader = self.dataset.make_loaders(
            workers=self.num_workers, batch_size=self.train_batch_size, 
            val_batch_size=self.test_batch_size, shuffle_val=False)
        self.train_ds = train_loader.dataset
        self.train_ds.transform = self.train_transform

        self.test_ds = test_loader.dataset
        self.test_ds.transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    
@utils.register_cc_dataset(name='imagenet')
class ImageNetCDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = './', batch_size=512, num_workers=4, 
        pin_memory=True, normalized=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.dims = (3, 224, 224)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.ToTensor(), T.Resize(256), T.CenterCrop(224), 
            T.Normalize(self.mean, self.std)])
        
        self.normalized = normalized
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
        ]

    def setup(
        self, stage: Optional[str] = None, severity_min: int = 1, severity_max: int = 6
    ):
        self.cc = {}

        for corruption in self.corruptions:
            in_corruption = []
            for severity in range(severity_min, severity_max):
                base_dataset = ImageNetC(self.data_dir, corruption, severity)
                _, test_loader = base_dataset.make_loaders(
                    self.num_workers, self.batch_size, only_val=True)
                inc_severity = test_loader.dataset
                inc_severity.transform = self.transform
                in_corruption.append(inc_severity)

            self.cc[corruption] = ConcatDataset(in_corruption)

    def test_dataloader(self):
        return {
            corruption: DataLoader(
                self.cc[corruption],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            for corruption in self.corruptions
        }


@utils.register_dataset(name='imagenet100')
class ImageNet100DataModule(ImageNetDataModule):
    def __init__(
        self, data_dir: str = './', train_batch_size=256, 
        test_batch_size=512, num_workers=4, pin_memory=True, 
        shuffle_train=True,
    ):
        super().__init__(
            data_dir, train_batch_size, test_batch_size, 
            num_workers, pin_memory, shuffle_train)
        self.num_classes = 100
        
    def setup(self, stage: Optional[str] = None):
        self.dataset = ImageNet100(self.data_dir)
        train_loader, test_loader = self.dataset.make_loaders(
            workers=self.num_workers, batch_size=self.train_batch_size, 
            val_batch_size=self.test_batch_size, shuffle_val=False)
        self.train_ds = train_loader.dataset
        self.train_ds.transform = self.train_transform

        self.test_ds = test_loader.dataset
        self.test_ds.transform = self.test_transform
    

@utils.register_cc_dataset(name='imagenet100')
class ImageNet100CDataModule(ImageNetCDataModule):

    def setup(
        self, stage: Optional[str] = None, severity_min: int = 1, severity_max: int = 6
    ):
        self.cc = {}

        for corruption in self.corruptions:
            in_corruption = []
            for severity in range(severity_min, severity_max):
                base_dataset = ImageNetC(self.data_dir, corruption, severity)
                _, test_loader = base_dataset.make_loaders(
                    self.num_workers, self.batch_size, only_val=True)
                inc_severity = test_loader.dataset
                inc_severity.transform = self.transform
                in_corruption.append(inc_severity)

            self.cc[corruption] = ConcatDataset(in_corruption)