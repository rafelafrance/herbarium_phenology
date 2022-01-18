"""Run a hydra model for training, testing, or inference."""
from argparse import Namespace
from typing import Union

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import db
from .hydra_dataset import HydraDataset

ArgsType = Union[Namespace]


class HydraRunner:
    """Base class for running a hydra model."""

    def __init__(self, model, traits, orders, args: ArgsType):
        self.model = model
        self.traits = traits
        self.orders = orders

        self.batch_size = args.batch_size
        self.database = args.database
        self.workers = args.workers
        self.limit = args.limit

        self.device = torch.device("cuda" if torch.has_cuda else "cpu")

    def get_dataset(self, database, split_run, split, augment=False, limit=0):
        """Get a dataset for training, validation, testing, or inference."""
        raw_data = db.select_split(database, split_run, split, limit)
        return HydraDataset(
            raw_data,
            self.model.backbone,
            traits=self.traits,
            orders=self.orders,
            augment=augment,
        )

    def to_device(self):
        """Move the model and other objects to the appropriate device."""
        self.model.to(self.device)


class HydraTrainingRunner(HydraRunner):
    """Train a hydra model."""

    def __init__(self, model, traits, orders, args: ArgsType):
        super().__init__(model, traits, orders, args)
        self.lr = args.learning_rate
        self.split_run = args.split_run

        self.train_dataset = self.get_dataset(
            self.database, self.split_run, "test", augment=True, limit=self.limit
        )
        self.val_dataset = self.get_dataset(
            self.database, self.split_run, "val", limit=self.limit
        )

        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizers = self.configure_optimizers()

    def configure_optimizers(self):
        """Configure the optimizer."""
        opts = []
        for _ in self.traits:
            opts.append(torch.optim.AdamW(self.model.parameters(), lr=self.lr))
        return opts

    def train_dataloader(self):
        """Load the training split for the data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            drop_last=len(self.train_dataset) % self.batch_size == 1,
        )

    def val_dataloader(self):
        """Load the validation split for the data."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.workers
        )

    def train(self, batch):
        """Train a model batch with images and orders."""

    @staticmethod
    def loss_function(y_pred, y_true):
        """Calculate the loss for all of the heads."""
        return F.binary_cross_entropy(y_pred, y_true)

    def run(self):
        """Train the model."""
        self.to_device()
        self.one_epoch(self.train_loader, self.optimizers)

    def one_epoch(self, loader, optimizers=None):
        """Train or validate an epoch."""
        from pprint import pp

        torch.autograd.set_detect_anomaly(True)
        avg = {t: {"loss": 0.0, "acc": 0.0, "n": 0.0} for t in self.traits}

        for images, orders, present, y_true, _ in loader:
            present = present.to(self.device)

            for i, trait in enumerate(self.traits):
                mask = present[:, i]
                true = y_true[:, i][mask]

                n = true.size()[0]
                if n > 0:
                    image = images[mask].to(self.device)
                    order = orders[mask].to(self.device)

                    pred = self.model(image, order)
                    pp(pred)
                    pp(true)

                    loss = self.loss_function(pred, true)

                    if optimizers:
                        optimizers[i].zero_grad()
                        loss.backward()
                        optimizers[i].step()

                    avg[trait]["loss"] += loss.item()
                    avg[trait]["acc"] += accuracy(pred, true)
                    avg[trait]["n"] += n

        for trait in self.traits:
            n = avg[trait]["n"]
            if n > 0.0:
                avg[trait]["loss"] /= n
                avg[trait]["acc"] /= n
        pp(avg)


class HydraTestingRunner(HydraRunner):
    """Test a hydra model."""


class HydraInferenceRunner(HydraRunner):
    """Run inference on a hydra model."""


def accuracy(y_pred, y_true):
    """Calculate the accuracy of the model."""
    pred = torch.round(y_pred)
    equals = (pred == y_true).type(torch.float)
    return torch.sum(equals)
