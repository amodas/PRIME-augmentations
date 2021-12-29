import os
import numpy as np

import torch
import pytorch_lightning as pl
import torchvision.datasets as dset
import torchvision.transforms as T

from absl import app, flags
from ml_collections.config_flags import config_flags
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from lightning.data import _DATASETS
from lightning.data import get_cc_dataset
from lightning.data import get_dataset

from models import AugModel, TransformLayer
from models import get_model, _MODELS

from lightning.systems import AugClassifier
from lightning.systems import Classifier

import utils
from utils.rand_filter import RandomFilter
from utils.color_jitter import RandomSmoothColor
from utils.diffeomorphism import Diffeo

from utils.augmix import AugMixDataset
from utils.prime import GeneralizedPRIMEModule
from utils.prime import PRIMEAugModule

from setup import setup_all, _setup


_WANDB_USERNAME = "username"
_WANDB_PROJECT = "common-corruptions"

def validate_config(cfg):
    if cfg.dataset not in _DATASETS:
        raise ValueError(f'Dataset {cfg.dataset} not supported!')
    elif cfg.model.name not in _MODELS[cfg.dataset]:
        raise ValueError(f'Model {cfg.model.name} not supported!')
    assert not (cfg.use_augmix and cfg.use_prime), 'Use only one augmentation!'
    if cfg.use_deepaugment:
        assert 'imagenet' in cfg.dataset, 'DeepAugment only supported on ImageNet!'
    
    if 'TMPDIR' in os.environ and 'imagenet' in cfg.dataset:
        setup_all(cfg.data_dir, cfg.cc_dir)
        if cfg.use_deepaugment:
            _setup(cfg.data_dir, 'EDSR')
            _setup(cfg.data_dir, 'CAE')
        cfg.data_dir = os.path.join(os.environ['TMPDIR'], cfg.dataset)
        cfg.cc_dir = os.path.join(os.environ['TMPDIR'], f'{cfg.dataset}c')


def main(_):
    config = flags.FLAGS.config
    validate_config(config)
    
    wandb.init(
        project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
        name=config.save_dir.split('/')[-2],
        settings=wandb.Settings(_disable_stats=True),
    )
    utils.print_config(config)
    wandb.config.update(config.to_dict())
    pl.seed_everything(1)
    
    
    # Setup train & val datasets
    dataset = get_dataset(config.dataset)(
        config.data_dir,
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        num_workers=config.train_num_workers,
    )
    
    transforms = [] if config.use_augmix else [T.ToTensor()]
    if 'imagenet' in config.dataset:
        transforms += [
            T.RandomResizedCrop(224), T.RandomHorizontalFlip()
        ]
        if not (config.use_prime or config.use_augmix):
            transforms.append(T.Normalize(dataset.mean, dataset.std))
        
        dataset.train_transform = T.Compose(transforms)
        dataset.test_transform = T.Compose([
            T.ToTensor(), T.Resize(256), T.CenterCrop(224),
            T.Normalize(dataset.mean, dataset.std)
        ])
    elif 'cifar' in config.dataset:
        transforms += [
            T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()
        ]
        if not (config.use_prime or config.use_augmix):
            transforms.append(T.Normalize(dataset.mean, dataset.std))
        
        dataset.train_transform = T.Compose(transforms)
        dataset.test_transform = T.Compose([
            T.ToTensor(), T.Normalize(dataset.mean, dataset.std)
        ])
        
    dataset.prepare_data()
    dataset.setup()

    
    # DeepAugment
    if config.use_deepaugment:
        edsr = dset.ImageFolder(
            os.path.join(dataset.data_dir, 'EDSR'), dataset.train_transform)
        cae = dset.ImageFolder(
            os.path.join(dataset.data_dir, 'CAE'), dataset.train_transform)
        dataset.train_ds = torch.utils.data.ConcatDataset(
            [dataset.train_ds, edsr, cae])
        dataset.train_ds.transform = dataset.train_transform
    
    
    # AugMix
    if config.use_augmix:
        preprocess = T.Compose([
            T.ToTensor(), T.Normalize(dataset.mean, dataset.std)])
        
        dataset.train_ds = AugMixDataset(
            dataset=dataset.train_ds, 
            preprocess=preprocess,
            mixture_width=config.augmix.mixture_width,
            mixture_depth=config.augmix.mixture_depth,
            aug_severity=config.augmix.severity,
            no_jsd=config.augmix.no_jsd,
            img_sz=dataset.dims[1]
        )
    
    
    # PRIME
    if config.use_prime:
        augmentations = []

        if config.enable_aug.diffeo:
            diffeo = Diffeo(
                sT=config.diffeo.sT, rT=config.diffeo.rT,
                scut=config.diffeo.scut, rcut=config.diffeo.rcut,
                cutmin=config.diffeo.cutmin, cutmax=config.diffeo.cutmax,
                alpha=config.diffeo.alpha, stochastic=True
            )
            augmentations.append(diffeo)

        if config.enable_aug.color_jit:
            color = RandomSmoothColor(
                cut=config.color_jit.cut, T=config.color_jit.T,
                freq_bandwidth=config.color_jit.max_freqs, stochastic=True
            )
            augmentations.append(color)

        if config.enable_aug.rand_filter:
            filt = RandomFilter(
                kernel_size=config.rand_filter.kernel_size,
                sigma=config.rand_filter.sigma, stochastic=True
            )
            augmentations.append(filt)
        
        prime_module = GeneralizedPRIMEModule(
            preprocess=TransformLayer(dataset.mean, dataset.std),
            mixture_width=config.augmix.mixture_width,
            mixture_depth=config.augmix.mixture_depth,
            no_jsd=config.augmix.no_jsd, max_depth=3,
            aug_module=PRIMEAugModule(augmentations),
        )
    
    
    # Setup model
    base_model = get_model(
        config.dataset, config.model.name
    )(num_classes=dataset.num_classes, pretrained=config.model.pretrained)
    
    if config.scheduler_t == "cyclic":
        config.lr_schedule.steps_per_epoch = int(np.ceil(len(dataset.train_ds) / config.train_batch_size))
    opt_cfg = {
        'optimizer_cfg': config.optimizer,
        'lr_schedule_cfg': config.lr_schedule,
        'scheduler_t': config.scheduler_t,
    }
    
    if config.use_prime:
        model = AugClassifier(
            model=AugModel(model=base_model, aug=prime_module),
            no_jsd=config.augmix.no_jsd, **opt_cfg,
        )
    elif config.use_augmix:
        model = AugClassifier(
            model=base_model, no_jsd=config.augmix.no_jsd, **opt_cfg,
        )
    else:
        model = Classifier(model=base_model, **opt_cfg)

    
    # PL trainer & logging
    wandb_logger = WandbLogger(project=_WANDB_PROJECT, entity=_WANDB_USERNAME, log_model=False)
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)
    checkpoint_callback = ModelCheckpoint(
        config.save_dir, filename='best', 
        monitor='val.acc', mode='max', save_last=True, 
        save_top_k=1, save_weights_only=True
    )
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=config.log_freq,
        gpus=-1,
        max_epochs=config.epochs,
        gradient_clip_val=config.grad_clip,
        callbacks=[lr_monitor, checkpoint_callback],
        num_sanity_val_steps=0,
        fast_dev_run=config.debug,
        accelerator=config.accelerator,
        benchmark=True
    )
    
    trainer.fit(model, dataset)
    model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
    if config.use_prime:
        del model.model.aug

    
    # Evaluation on common corruptions
    dataset_c = get_cc_dataset(config.dataset)(
        config.cc_dir, batch_size=config.test_batch_size,
        num_workers=config.test_num_workers
    )
    transforms = [] if 'cifar' in config.dataset else [T.ToTensor()]
    transforms.append(T.Normalize(dataset.mean, dataset.std))
    dataset_c.transform = T.Compose(transforms)
    
    dataset_c.prepare_data()
    dataset_c.setup()

    cc_loaders = dataset_c.test_dataloader()
    keys = list(cc_loaders.keys())
    avg_acc = 0.
    for key in keys:
        res = trainer.test(model, test_dataloaders=cc_loaders[key])
        acc = res[0]["test.acc"]
        wandb.run.summary["test.%s" % key] = acc
        avg_acc += acc
    wandb.run.summary["test_avg.acc"] = avg_acc / len(keys)
    wandb.run.summary["val.acc"] = checkpoint_callback.best_model_score


if __name__ == "__main__":
    config_flags.DEFINE_config_file('config')
    app.run(main)
