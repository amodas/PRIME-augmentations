import numpy as np
import torch

from PIL import Image, ImageOps, ImageEnhance


IMAGE_SIZE = 224  # Code should change this value

class AugMixDataset(torch.utils.data.Dataset):

    def __init__(
        self, dataset, preprocess, all_ops=False, mixture_width=3, 
        mixture_depth=-1, aug_severity=3, no_jsd=False, aug_list=None, 
        img_sz=32
    ):
        """
        Dataset wrapper to perform AugMix augmentation.

        :param dataset: A torch.Dataset (i.e., CIFAR10/100) object
        :param preprocess: Preprocessing function which should return a torch tensor
        :param all_ops: Weather to use all augmentation operations (including the forbidden ones such as brightness)
        :param mixture_width: Number of augmentation chains to mix per augmented example
        :param mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
        :param aug_severity: Severity of base augmentation operators
        :param no_jsd: Turn off JSD consistency loss
        """

        self.dataset = dataset
        self.preprocess = preprocess
        if aug_list is None:
            self.aug_list = augmentations if not all_ops else augmentations_all
        else:
            self.aug_list = aug_list
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.no_jsd = no_jsd

        global IMAGE_SIZE
        IMAGE_SIZE = img_sz

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), self.aug(x, self.preprocess),
                        self.aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

    def aug(self, image, preprocess):

        ws = np.float32(np.random.dirichlet([1] * self.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity)

            # Preprocessing commutes since all coefficients are convex
            mix = mix + ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix

        return mixed


#################################################
# Here starts a list of different augmentations #
#################################################


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR
    )


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR
    )


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR
    )


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR
    )


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

augmentations = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

augmentations_all = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]