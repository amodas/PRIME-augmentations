import torch

from torch.distributions import Dirichlet, Beta
from einops import rearrange, repeat
from opt_einsum import contract


class PRIMEAugModule(torch.nn.Module):
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = augmentations
        self.num_transforms = len(augmentations)

    def forward(self, x, mask_t):
        aug_x = torch.zeros_like(x)
        for i in range(self.num_transforms):
            aug_x += self.augmentations[i](x) * mask_t[:, i]
        return aug_x

class GeneralizedPRIMEModule(torch.nn.Module):
    def __init__(
        self, preprocess, aug_module, mixture_width=3, 
        mixture_depth=-1, no_jsd=False, max_depth=3
    ):
        """
        Wrapper to perform PRIME augmentation.

        :param preprocess: Preprocessing function which should return a torch tensor
        :param all_ops: Weather to use all augmentation operations (including the forbidden ones such as brightness)
        :param mixture_width: Number of augmentation chains to mix per augmented example
        :param mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
        :param no_jsd: Turn off JSD consistency loss
        """
        super().__init__()
        self.preprocess = preprocess
        self.aug_module = aug_module
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.no_jsd = no_jsd

        self.max_depth = max_depth
        self.depth = self.mixture_depth if self.mixture_depth > 0 else self.max_depth
        self.depth_combos = torch.tril(torch.ones((max_depth, max_depth)))

    @torch.no_grad()
    def forward(self, img):
        if self.no_jsd:
            return self.aug(img)
        else:
            return torch.cat((self.preprocess(img), self.aug(img),
                              self.aug(img)), 0)

    
    def aug(self, img):
        self.dirichlet = Dirichlet(concentration=torch.tensor([1.] * self.mixture_width, device=img.device))
        self.beta = Beta(concentration1=torch.ones(1, device=img.device, dtype=torch.float32), concentration0=torch.ones(1, device=img.device, dtype=torch.float32))

        ws = self.dirichlet.sample([img.shape[0]])
        m = self.beta.sample([img.shape[0]])[..., None, None]

        img_repeat = repeat(img, 'b c h w -> m b c h w', m=self.mixture_width)
        img_repeat = rearrange(img_repeat, 'm b c h w -> (m b) c h w')

        trans_combos = torch.eye(self.aug_module.num_transforms, device=img_repeat.device)
        depth_mask = torch.zeros(img_repeat.shape[0], self.max_depth, 1, 1, 1, device=img_repeat.device)
        trans_mask = torch.zeros(img_repeat.shape[0], self.aug_module.num_transforms, 1, 1, 1, device=img_repeat.device)

        depth_idx = torch.randint(0, len(self.depth_combos), size=(img_repeat.shape[0],))
        depth_mask.data[:, :, 0, 0, 0] = self.depth_combos[depth_idx]

        image_aug = img_repeat.clone()

        for d in range(self.depth):

            trans_idx = torch.randint(0, len(trans_combos), size=(img_repeat.shape[0],))
            trans_mask.data[:, :, 0, 0, 0] = trans_combos[trans_idx]

            image_aug.data = depth_mask[:, d] * self.aug_module(image_aug, trans_mask) + (1. - depth_mask[:, d]) * image_aug

        image_aug = rearrange(self.preprocess(image_aug), '(m b) c h w -> m b c h w', m=self.mixture_width)

        mix = torch.einsum('bm, mbchw -> bchw', ws, image_aug)
        mixed = (1. - m) * self.preprocess(img) + m * mix

        return mixed