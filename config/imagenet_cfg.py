import os
import ml_collections


def get_config():
    num_gpus = 2 # for linear scaling

    config = ml_collections.ConfigDict()
    config.optimizer = ml_collections.ConfigDict()
    config.lr_schedule = ml_collections.ConfigDict()

    config.diffeo = ml_collections.ConfigDict()
    config.color_jit = ml_collections.ConfigDict()
    config.rand_filter = ml_collections.ConfigDict()
    config.augmix = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()
    
    # Dataset
    config.dataset = 'imagenet'
    config.data_dir = './data/imagenet/'
    config.cc_dir = './data/imagenetc/'
    config.train_batch_size = 256 * num_gpus
    config.test_batch_size = 512 * num_gpus
    config.train_num_workers = 36
    config.test_num_workers = 36

    # Model
    config.save_dir = './PRIME/in_prime/'
    config.model.name = 'resnet50'
    config.model.pretrained = True    
    config.accelerator = 'dp'
    
    # Training
    epochs = ml_collections.FieldReference(100)
    config.epochs = epochs
    config.grad_clip = 0.0
    
    lr = ml_collections.FieldReference(0.01) * num_gpus
    config.optimizer.lr = lr
    config.optimizer.momentum = 0.9
    config.optimizer.weight_decay = 1e-4
    config.optimizer.nesterov = True
    
    config.scheduler_t = 'cyclic'
    config.lr_schedule.max_lr = lr
    config.lr_schedule.pct_start = 0.1
    config.lr_schedule.epochs = epochs
    config.lr_schedule.steps_per_epoch = 391

    # Augmentations
    config.diffeo.sT = 1.
    config.diffeo.rT = 1.
    config.diffeo.scut = 1.
    config.diffeo.rcut = 1.
    config.diffeo.cutmin = 2
    config.diffeo.cutmax = 500
    config.diffeo.alpha = 1.0

    config.color_jit.cut = 500
    config.color_jit.T = 0.05
    config.color_jit.max_freqs = 20

    config.rand_filter.kernel_size = 3
    config.rand_filter.sigma = 4.0
    
    config.enable_aug = ml_collections.ConfigDict()
    config.enable_aug.diffeo = True
    config.enable_aug.color_jit = True
    config.enable_aug.rand_filter = True
    
    config.augmix.all_ops = False
    config.augmix.mixture_width = 3
    config.augmix.mixture_depth = -1
    config.augmix.no_jsd = True
    config.augmix.severity = 1
    
    config.use_augmix = False
    config.use_deepaugment = False
    config.use_prime = True
    
    config.log_freq = 50
    config.debug = False
    return config