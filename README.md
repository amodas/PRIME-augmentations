# PRIME: A Few Primitives Can Boost Robustness to Common Corruptions

PRIME is a general data augmentation scheme that consists of simple families of max-entropy image transformations for conferring robustness to common corruptions.
<p align="center">
    <img src="misc/prime-augmentations.png"/>
</p>

### Setup

This code has been tested with `Python 3.8.5` and `PyTorch 1.9.1`. To install required dependencies run:
```sh
$ pip install -r requirements.txt
```
For evaluation, download and extract the CIFAR-10-C, CIFAR-100-C and ImageNet-C datasets from [here](https://github.com/hendrycks/robustness).

### Usage

Train ResNet-50 on ImageNet with PRIME:
```sh
$ python -u train.py --config=config/imagenet_cfg.py \
    --config.save_dir=<save_dir> \
    --config.data_dir=<data_dir> \
    --config.cc_dir=<common_corr_dir> \
    --config.use_prime=True
```
Other configuration options can be found in [`config`](config/).

### Results

### Citing this work

```
```