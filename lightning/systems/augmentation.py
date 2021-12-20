import flatten_dict
import ml_collections
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from lightning.systems.classification import Classifier


class AugClassifier(Classifier):
    def __init__(
        self,
        model: nn.Module = None,
        optimizer_cfg: ml_collections.ConfigDict = None,
        lr_schedule_cfg: ml_collections.ConfigDict = None,
        no_jsd: bool = False,
        scheduler_t: str = "cyclic",
        test_keys=None,
    ):
        super().__init__(
            model, optimizer_cfg, lr_schedule_cfg, 
            scheduler_t=scheduler_t, test_keys=test_keys)
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.loss_ema = 0.0
        self.no_jsd = no_jsd

    def training_step(self, batch, batch_idx):

        if self.no_jsd:
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, 1)

            loss = F.cross_entropy(logits, y)
            return {"loss": loss, "preds": preds, "targets": y}

        else:
            x, y = batch
            if len(batch[0]) == 3: # for AugMix
                x = torch.cat(x, 0)
            
            logits_all = self(x)
            if len(batch[0]) == 3: # for AugMix
                logits, logits_aug1, logits_aug2 = torch.split(
                    logits_all, batch[0][0].size(0))
            else:
                logits, logits_aug1, logits_aug2 = torch.split(
                    logits_all, x.size(0))

            preds = torch.argmax(logits, 1)
            loss = F.cross_entropy(logits, y)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits, dim=1), F.softmax(
                logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            self.loss_ema = self.loss_ema * 0.9 + loss * 0.1
            return {"loss": loss, "preds": preds, "targets": y, "loss_ema": self.loss_ema}

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]
        losses = batch_parts["loss"]

        loss = losses.mean()
        self.train_acc(preds, targets)
        self.log("train.acc", self.train_acc, on_step=True)
        self.log("train.loss", loss, on_step=True)
        if not self.no_jsd:
            self.log("train.loss_ema", batch_parts["loss_ema"].mean(), on_step=True)

        return loss