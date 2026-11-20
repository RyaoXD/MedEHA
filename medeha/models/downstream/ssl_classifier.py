from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy


class SSLFineTuner(LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 model_name: str = "resnet_50",
                 in_features: int = 2048,
                 num_classes: int = 5,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 multilabel: bool = True,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if multilabel:
            if num_classes == 1:
                self.val_auc = AUROC(task="binary", num_classes=num_classes)
                self.test_auc = AUROC(task="binary", num_classes=num_classes)
            else:
                self.val_auc = AUROC(task="multilabel", num_classes=num_classes)
                self.test_auc = AUROC(task="multilabel", num_classes=num_classes)
        else:
            if num_classes == 1:
                self.val_acc = Accuracy(
                    task="binary", num_classes=num_classes, top_k=1)
                self.test_acc = Accuracy(
                    task="binary", num_classes=num_classes, top_k=1)
            else:
                self.val_acc = Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=1)
                self.test_acc = Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=1)

        self.save_hyperparameters(ignore=['backbone'])

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.model_name = model_name
        self.multilabel = multilabel

        self.linear_layer = SSLEvaluator(
            n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            self.val_auc.update(torch.sigmoid(logits).float(), y.long())
        else:
            self.val_acc.update(F.softmax(logits, dim=-1).float(), y.long())

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        if self.multilabel:
            if validation_step_outputs:
                self.log("val_auc", self.val_auc.compute(), prog_bar=True,
                         on_epoch=True, sync_dist=True)
            self.val_auc.reset()
        else:
            if validation_step_outputs:
                self.log("val_acc", self.val_acc.compute(), prog_bar=True,
                         on_epoch=True, sync_dist=True)
            self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        if self.multilabel:
            self.test_auc.update(torch.sigmoid(logits).float(), y.long())
        else:
            self.test_acc.update(F.softmax(logits, dim=-1).float(), y.long())

        return loss

    def test_epoch_end(self, test_step_outputs):
        if self.multilabel:
            if test_step_outputs:
                self.log("test_auc", self.test_auc.compute(), on_epoch=True)
            self.test_auc.reset()
        else:
            if test_step_outputs:
                self.log("test_acc", self.test_acc.compute(), on_epoch=True)
            self.test_acc.reset()

    def shared_step(self, batch):
        x, y = batch
        # For multi-class
        with torch.no_grad():
            feats, _, _ = self.backbone(x, None)
            # if "vit" in self.model_name:
            #     feats = feats[:, 0]
        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), y.float())
        else:
            y = y.squeeze()
            loss = F.cross_entropy(logits.float(), y.long())

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
