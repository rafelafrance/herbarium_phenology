"""Create EfficientNets that uses plant orders as well as images for input."""
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from . import db
from .herbarium_dataset import HerbariumDataset

MODELS = {
    "b0": {
        "model": torchvision.models.efficientnet_b0,
        "size": (224, 224),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b1": {
        "model": torchvision.models.efficientnet_b1,
        "size": (240, 240),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b2": {
        "model": torchvision.models.efficientnet_b2,
        "size": (260, 260),
        "dropout": 0.3,
        "in_feat": 1408,
    },
    "b3": {
        "model": torchvision.models.efficientnet_b3,
        "size": (300, 300),
        "dropout": 0.3,
        "in_feat": 1536,
    },
    "b4": {
        "model": torchvision.models.efficientnet_b4,
        "size": (380, 380),
        "dropout": 0.4,
        "in_feat": 1792,
    },
    # b5: {"size": (456, 456), }
    # b6: {"size": (528, 528), }
    "b7": {
        "model": torchvision.models.efficientnet_b7,
        "size": (600, 600),
        "dropout": 0.5,
        "in_feat": 2560,
    },
}


class EfficientNetOrders(pl.LightningModule):
    """An EfficientNet that uses phylogenetic orders as sidecar data."""

    def __init__(self, orders: list[str], args: dict[Any]):
        super().__init__()

        self.orders = orders
        self.net = args["net"]
        self.workers = args["workers"]
        self.database = args["database"]
        self.split_run = args["split_run"]
        self.batch_size = args["batch_size"]
        self.lr = args["learning_rate"]
        self.limit = args["limit"]

        # TODO Do I still need these?
        self.epochs = args["epochs"]
        self.save_model = args["save_model"]

        params = MODELS[args["net"]]
        self.size = params["size"]
        self.mean = (0.485, 0.456, 0.406)  # ImageNet
        self.std_dev = (0.229, 0.224, 0.225)  # ImageNet

        # pos_weight = train_dataset.pos_weight()
        self.criterion = nn.BCEWithLogitsLoss()

        fc_feat1 = params["in_feat"] // 4
        fc_feat2 = params["in_feat"] // 8
        fc_feat3 = params["in_feat"] // 16
        mix_feat = fc_feat1 + len(orders)
        out_feat = len(HerbariumDataset.all_traits)

        self.backbone = params["model"](pretrained=True)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=params["in_feat"], out_features=fc_feat1),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat1),
        )

        self.multi_classifier = nn.Sequential(
            nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=mix_feat, out_features=fc_feat2),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat2),
            #
            nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=fc_feat2, out_features=fc_feat3),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat3),
            #
            # nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(in_features=fc_feat3, out_features=out_feat),
            # nn.Sigmoid(),
        )

        self.state = torch.load(args["load_weights"]) if args["load_weights"] else {}
        if self.state.get("model_state"):
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0, x1):
        """Run the classifier forwards."""
        x0 = self.backbone(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.multi_classifier(x)
        return x

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        """Load the training split for the data."""
        split = db.select_split(
            self.database, self.split_run, split="train", limit=self.limit
        )
        dataset = HerbariumDataset(split, self, orders=self.orders, augment=True)
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            drop_last=len(split) % self.batch_size == 1,
        )
        return loader

    def val_dataloader(self):
        """Load the validation split for the data."""
        split = db.select_split(
            self.database,
            self.split_run,
            split="val",
            limit=self.limit,
        )
        dataset = HerbariumDataset(split, self, orders=self.orders)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.workers
        )
        return loader

    def training_step(self, batch, _):
        """Train a model batch with images and orders."""
        print(_)
        images, orders, y_true, _ = batch
        y_pred = self(images, orders)
        loss = self.criterion(y_pred, y_true)
        # self.log(
        #     "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
        #     logger=True
        # )
        return loss

    def validation_step(self, batch, _):
        """Validate a model batch with images and orders."""
        images, orders, y_true, _ = batch
        y_pred = self(images, orders)
        loss = self.criterion(y_pred, y_true)
        # self.log(
        #     "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        # )
        return loss
