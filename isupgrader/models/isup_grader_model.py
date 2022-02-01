from typing import Any
from pandas.core.frame import DataFrame
import os
from pathlib import Path
from glob import glob

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score


class ISUPGraderModel(pl.LightningModule):
    def __init__(self,
                 net: nn.Module,
                 criterion: Any,
                 lr: float,
                 weight_decay: float,
                 n_epochs: int,
                 datamodule: pl.LightningDataModule = None):
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.n_epochs = n_epochs

        self.save_hyperparameters(ignore=["net", "datamodule"])

        self.automatic_optimization = True

        self.datamodule = datamodule

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return int(((batches // effective_accum) * self.trainer.max_epochs) * 1.1)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.num_training_steps,
            pct_start=self.steps_per_epoch/self.num_training_steps,
            div_factor=10,
        )

        scheduler = {
            'scheduler': sched,
            'interval': 'step',
        }

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        try:
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()

    def test_step(self, batch, batch_idx):
        data, targets, slide_ids = batch

        logits = self.forward(data)

        loss = self.criterion(logits, targets)

        preds = logits.detach().clamp(0, 5).round()

        accuracy = (preds.round() == targets).float().mean()

        return {"test_loss": loss, "test_acc": accuracy, "test_preds": preds, "test_targets": targets, "test_slide_ids": slide_ids}

    def training_step(self, batch, batch_idx):
        # Output from Dataloader
        data, targets, _ = batch

        logits = self.forward(data)

        loss = self.criterion(logits, targets)

        preds = logits.detach().clamp(0, 5).round()

        accuracy = (preds == targets).float().mean()

        self.log("train_step_loss", loss)
        self.log("train_step_acc", accuracy)
        self.log("train_step_lr", self.optimizers().param_groups[0]["lr"])

        return {"loss": loss, "acc": accuracy, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        # Output from Dataloader
        data, targets, _ = batch

        logits = self.forward(data)

        loss = self.criterion(logits, targets)

        preds = logits.detach().clamp(0, 5).round()

        accuracy = (preds == targets).float().mean()

        self.log("val_step_loss", loss, sync_dist=True)
        self.log("val_step_acc", accuracy, sync_dist=True)

        return {"val_loss": loss, "val_acc": accuracy, "val_preds": preds, "val_targets": targets}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).float().mean()
        avg_accuracy = torch.stack([x["acc"] for x in outputs]).float().mean()

        preds_out = [x["preds"].cpu().numpy() for x in outputs]
        targets_out = [x["targets"].cpu().numpy() for x in outputs]

        preds = []
        targets = []

        for pred, target in zip(preds_out, targets_out):
            preds.extend([int(x.item()) for x in pred])
            targets.extend([int(x.item()) for x in target])

        qwk = cohen_kappa_score(preds, targets, weights='quadratic')

        self.log("train_epoch_loss", avg_loss)
        self.log("train_epoch_acc", avg_accuracy)
        self.log("train_epoch_qwk", qwk)
        self.log("train_epoch_lr", self.optimizers().param_groups[0]["lr"])

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).float().mean()
        avg_accuracy = torch.stack([x["val_acc"] for x in outputs]).float().mean()

        preds_out = [x["val_preds"].cpu().numpy() for x in outputs]
        targets_out = [x["val_targets"].cpu().numpy() for x in outputs]

        preds = []
        targets = []

        for pred, target in zip(preds_out, targets_out):
            preds.extend([int(x.item()) for x in pred])
            targets.extend([int(x.item()) for x in target])

        qwk = cohen_kappa_score(preds, targets, weights='quadratic')

        self.log("val_epoch_loss", avg_loss)
        self.log("val_epoch_acc", avg_accuracy)
        self.log("val_epoch_qwk", qwk)

        return {"avg_val_loss": avg_loss, "avg_val_acc": avg_accuracy, "avg_val_qwk": qwk}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).float().mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).float().mean()
        preds_out = [x["test_preds"].cpu().numpy() for x in outputs]
        targets_out = [x["test_targets"].cpu().numpy() for x in outputs]
        slide_ids_out = [x["test_slide_ids"] for x in outputs]

        preds = []
        preds_raw = []
        targets = []
        slide_ids = []

        for pred, target, slide_id in zip(preds_out, targets_out, slide_ids_out):
            preds_raw.extend([x.item() for x in pred])
            preds.extend([int(round(x.item())) for x in pred])
            targets.extend([int(x.item()) for x in target])
            slide_ids.extend(slide_id)

        qwk = cohen_kappa_score(preds, targets, weights='quadratic')
        if self.trainer.is_global_zero:
            self.log("test_epoch_loss", avg_loss, rank_zero_only=True)
            self.log("test_epoch_acc", avg_accuracy, rank_zero_only=True)
            self.log("test_epoch_qwk", qwk, rank_zero_only=True)

        return {"avg_test_loss": avg_loss, "avg_test_acc": avg_accuracy, "avg_test_qwk": qwk}

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def setup(self, stage=None) -> None:
        self.datamodule.setup(stage)