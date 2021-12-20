import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive

from . import utils


class TransformedTensorDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor, transform=None) -> None:
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            image, label = super().__getitem__(index)
            return self.transform(image), label
        else:
            return super().__getitem__(index)


@utils.register_dataset(name='cifar10')
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = './', train_batch_size=128,
        test_batch_size=256, num_workers=4, pin_memory=True,
    ):
        
        super().__init__()
        self.data_dir = data_dir
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2471, 0.2435, 0.2616]
        self.num_classes = 10
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = T.Compose([
            T.ToTensor(), T.Normalize(self.mean, self.std)])
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dims = (3, 32, 32)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        self.train_ds = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.train_transform
        )
        self.test_ds = datasets.CIFAR10(
            self.data_dir, train=False, transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
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


@utils.register_cc_dataset(name='cifar10')
class CIFAR10CDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = './', batch_size=256, num_workers=4, 
        pin_memory=True, normalized=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.dims = (3, 32, 32)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"  # URL FROM ZENODO

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

    @property
    def dataset_path(self):
        return self.data_dir
    
    def prepare_data(self):
        if not os.path.exists(self.dataset_path):
            download_and_extract_archive(
                self.url,
                download_root=self.data_dir,
                filename=self.dirname + ".tar",
                remove_finished=True,
            )
            self.data_dir = os.path.join(self.data_dir, self.dirname)
        else:
            print("Files already downloaded and verified")

    def setup(self, stage: Optional[str] = None):
        self.cc = {}
        labels = torch.tensor(np.load(os.path.join(self.dataset_path, "labels.npy")), dtype=torch.int64)

        for corruption in self.corruptions:
            raw_imgs = np.load(os.path.join(self.dataset_path, corruption + ".npy"))
            images = raw_imgs.transpose([0, 3, 1, 2])
            if self.normalized:
                images = images.astype(np.float32) / 255.0

            self.cc[corruption] = TransformedTensorDataset(
                torch.tensor(images), labels, transform=self.transform
            )

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
    

@utils.register_dataset(name='cifar100')
class CIFAR100DataModule(CIFAR10DataModule):
    def __init__(
        self, data_dir: str = './', train_batch_size=128,
        test_batch_size=256, num_workers=4, pin_memory=True,
    ):
        super().__init__(
            data_dir, train_batch_size, test_batch_size, 
            num_workers, pin_memory)
        self.mean = [0.5071, 0.4865, 0.4409]
        self.std = [0.2673, 0.2564, 0.2762]
        self.num_classes = 100
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = T.Compose([
            T.ToTensor(), T.Normalize(self.mean, self.std)])
        
    def prepare_data(self):
        # download
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        self.train_ds = datasets.CIFAR100(
            self.data_dir, train=True, transform=self.train_transform
        )
        self.test_ds = datasets.CIFAR100(
            self.data_dir, train=False, transform=self.test_transform
        )


@utils.register_cc_dataset(name='cifar100')
class CIFAR100CDataModule(CIFAR10CDataModule):
    def __init__(
        self, data_dir: str='./', batch_size=256, num_workers=4, 
        pin_memory=True, normalized=True
    ):
        super().__init__(
            data_dir, batch_size, num_workers, pin_memory, normalized)
        self.dirname = "CIFAR-100-C"
        self.transform = None
        self.url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"  # URL FROM ZENODO