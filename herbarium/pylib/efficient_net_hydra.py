"""EfficientNet using plant orders and images for input, separate heads per trait."""
from abc import ABC
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from . import db
from .herbarium_dataset import HerbariumDataset

BACKBONES = {
    "b0": {
        "backbone": torchvision.models.efficientnet_b0,
        "size": (224, 224),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b1": {
        "backbone": torchvision.models.efficientnet_b1,
        "size": (240, 240),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b2": {
        "backbone": torchvision.models.efficientnet_b2,
        "size": (260, 260),
        "dropout": 0.3,
        "in_feat": 1408,
    },
    "b3": {
        "backbone": torchvision.models.efficientnet_b3,
        "size": (300, 300),
        "dropout": 0.3,
        "in_feat": 1536,
    },
    "b4": {
        "backbone": torchvision.models.efficientnet_b4,
        "size": (380, 380),
        "dropout": 0.4,
        "in_feat": 1792,
    },
    # b5: {"size": (456, 456), }
    # b6: {"size": (528, 528), }
    "b7": {
        "backbone": torchvision.models.efficientnet_b7,
        "size": (600, 600),
        "dropout": 0.5,
        "in_feat": 2560,
    },
}


class EfficientNetHydra(pl.LightningModule, ABC):
    """An EfficientNet that uses phylogenetic orders as sidecar data."""

    def __init__(self, traits: list[str], orders: list[str], args: dict[Any]):
        super().__init__()

        self.traits = traits
        self.orders = orders
        self.net = args["net"]
        self.workers = args["workers"]
        self.database = args["database"]
        self.split_run = args["split_run"]
        self.batch_size = args["batch_size"]
        self.lr = args["learning_rate"]

        params = BACKBONES[args["backbone"]]
        self.size = params["size"]
        self.mean = (0.485, 0.456, 0.406)  # ImageNet
        self.std_dev = (0.229, 0.224, 0.225)  # ImageNet

        in_feat = params["in_feat"] + len(orders)
        fc_feat1 = in_feat // 4
        fc_feat2 = in_feat // 8
        fc_feat3 = in_feat // 16

        self.backbone = params["backbone"](pretrained=True)
        self.backbone.classifier = nn.Sequential(nn.Identity())

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.heads = []
        for _ in traits:
            self.heads.append(
                nn.Sequential(
                    nn.Dropout(p=params["dropout"], inplace=True),
                    nn.Linear(in_features=in_feat, out_features=fc_feat1),
                    nn.SiLU(inplace=True),
                    nn.BatchNorm1d(num_features=fc_feat1),
                    #
                    nn.Dropout(p=params["dropout"], inplace=True),
                    nn.Linear(in_features=fc_feat1, out_features=fc_feat2),
                    nn.SiLU(inplace=True),
                    nn.BatchNorm1d(num_features=fc_feat2),
                    #
                    nn.Dropout(p=params["dropout"], inplace=True),
                    nn.Linear(in_features=fc_feat2, out_features=fc_feat3),
                    nn.SiLU(inplace=True),
                    nn.BatchNorm1d(num_features=fc_feat3),
                    #
                    # nn.Dropout(p=self.dropout, inplace=True),
                    nn.Linear(in_features=fc_feat3, out_features=1),
                    # nn.Sigmoid(),
                )
            )

        self.state = torch.load(args["load_weights"]) if args["load_weights"] else {}
        if self.state.get("model_state"):
            self.load_state_dict(self.state["model_state"])

        self.train_dataset = self.get_dataset("train", augment=True)

        self._pos_weight = self.train_dataset.pos_weight()
        self.pos_weight = None

        # Important: This property activates manual optimization
        self.automatic_optimization = False

    def forward(self, x0, x1):
        """Run the classifier forwards through multiple heads."""
        x0 = self.backbone(x0)
        x = torch.cat((x0, x1), dim=1)
        return [h(x) for h in self.heads]

    def loss_function(self, y_pred, y_true):
        """Calculate the loss for each head."""
        return F.binary_cross_entropy_with_logits(
            y_pred, y_true, pos_weight=self.pos_weight
        )

    def configure_optimizers(self):
        """Configure the optimizers for all heads."""
        self.pos_weight = torch.tensor(self._pos_weight, device=self.device)
        return [torch.optim.AdamW(h.parameters(), lr=self.lr) for h in self.heads]

    def get_dataset(self, split, augment=False):
        """Get a dataset for training, validation, testing, or inference."""
        raw_data = db.select_split(self.database, self.split_run, split=split)
        return HerbariumDataset(raw_data, self, orders=self.orders, augment=augment)

    def train_dataloader(self):
        """Load the training split for the data."""
        dataset = self.get_dataset("train", augment=True)
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            drop_last=len(self.train_dataset) % self.batch_size == 1,
        )

    def val_dataloader(self):
        """Load the validation split for the data."""
        dataset = self.get_dataset("val")
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers)

    def training_step(self, batch, _):
        """Train a model batch with images and orders."""
        images, orders, y_true, _ = batch
        y_pred = self(images, orders)

        losses = []
        for true, pred, opt in zip(y_true, y_pred, self.optimizers()):
            losses.append(self.loss_function(pred, true))

        loss = self.loss_function(y_pred, y_true)
        return {"loss": loss}

    def validation_step(self, batch, _):
        """Validate a model batch with images and orders."""
        images, orders, y_true, _ = batch
        y_pred = self(images, orders)
        loss = self.loss_function(y_pred, y_true)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        """Log epoch results."""
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"avg_val_loss": val_loss}
        return {"log": log}


def accuracy(y_pred, y_true):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(y_pred))
    equals = (pred == y_true).type(torch.float)
    return torch.mean(equals)
