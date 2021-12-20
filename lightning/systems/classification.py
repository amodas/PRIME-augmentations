import flatten_dict
import ml_collections
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class Classifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module = None,
        optimizer_cfg: ml_collections.ConfigDict = None,
        lr_schedule_cfg: ml_collections.ConfigDict = None,
        scheduler_t: str = "cyclic",
        test_keys=None,
    ):
        super().__init__()
        self.model = model

        self.optimizer_cfg = optimizer_cfg
        self.lr_schedule_cfg = lr_schedule_cfg
        self.scheduler_t = scheduler_t

        self.train_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.val_acc = torchmetrics.Accuracy(dist_sync_on_step=True)

        if test_keys is None:
            self.test_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        else:
            self.test_acc = nn.ModuleDict({key: torchmetrics.Accuracy() for key in test_keys})

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_cfg)
        if self.scheduler_t == "cyclic":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.lr_schedule_cfg),
                "interval": "step",
            }
        elif self.scheduler_t == "piecewise_constant":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_schedule_cfg),
                "interval": "epoch",
            }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)

        loss = F.cross_entropy(logits, y)
        return {'loss': loss, 'preds': preds, 'targets': y}

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]
        losses = batch_parts["loss"]

        loss = losses.mean()
        self.train_acc(preds, targets)
        self.log("train.acc", self.train_acc, on_step=True)
        self.log("train.loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)
        return {"preds": preds, "targets": y}

    def validation_step_end(self, batch_parts):
        preds=batch_parts["preds"]
        targets=batch_parts["targets"]

        self.val_acc(preds, targets)
        self.log("val.acc", self.val_acc, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, dict):
            preds = {}
            for key, (x, y) in flatten_dict.flatten(batch, reducer="dot").items():
                logits = self(x)
                preds[key] = torch.argmax(logits, 1)

            return {"targets": y, **preds}
        else:
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, 1)
            return {"preds": preds, "targets": y}

    def test_step_end(self, batch_parts):
        if len(batch_parts.keys()) > 2:
            targets = batch_parts.pop("targets")
            avg_acc = []
            for key, preds in batch_parts.items():
                cur_acc = self.test_acc[key](preds, targets)
                self.log(f"test.{key}", self.test_acc[key], on_step=False, on_epoch=True)
                avg_acc.append(cur_acc)

            self.log("test_avg.acc", torch.tensor(avg_acc).mean(), on_step=False, on_epoch=True)
        else:
            preds = batch_parts["preds"]
            targets = batch_parts["targets"]
            self.test_acc(preds, targets)
            self.log("test.acc", self.test_acc, on_step=False, on_epoch=True)
