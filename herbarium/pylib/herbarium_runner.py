"""Run a hydra model for training, testing, or inference."""
import logging
from abc import ABC
from abc import abstractmethod
from argparse import Namespace
from typing import Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from . import db
from . import log
from .herbarium_dataset import HerbariumDataset
from .herbarium_dataset import InferenceDataset

ArgsType = Union[Namespace]


class HerbariumRunner(ABC):
    """Base class for running a hydra model."""

    def __init__(self, model, orders, args: ArgsType):
        self.model = model
        self.orders = orders

        self.trait = args.trait
        self.batch_size = args.batch_size
        self.database = args.database
        self.workers = args.workers
        self.limit = args.limit

        self.device = torch.device("cuda" if torch.has_cuda else "cpu")
        self.model.to(self.device)

    def get_dataset(
        self, database, split_set, split, target_set, trait, augment=False, limit=0
    ):
        """Get a dataset for training, validation, testing, or inference."""
        raw_data = db.select_split(
            database=database,
            split_set=split_set,
            split=split,
            target_set=target_set,
            trait=trait,
            limit=limit,
        )
        return HerbariumDataset(
            raw_data,
            self.model.backbone,
            orders=self.orders,
            augment=augment,
        )

    @abstractmethod
    def run(self):
        """Run the function of the class"""


class HerbariumTrainingRunner(HerbariumRunner):
    """Train a hydra model."""

    def __init__(self, model, orders, args: ArgsType):
        super().__init__(model, orders, args)

        self.lr = args.learning_rate
        self.split_set = args.split_set
        self.save_model = args.save_model
        self.target_set = args.target_set

        self.train_dataset = self.get_dataset(
            database=self.database,
            split_set=self.split_set,
            split="train",
            target_set=self.target_set,
            trait=self.trait,
            augment=True,
            limit=self.limit,
        )
        self.val_dataset = self.get_dataset(
            database=self.database,
            split_set=self.split_set,
            split="val",
            target_set=self.target_set,
            trait=self.trait,
            limit=self.limit,
        )

        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer = self.configure_optimizers()
        self.criterion = self.configure_criteria()

        self.best_loss = self.model.state.get("best_loss", np.Inf)
        self.best_acc = self.model.state.get("accuracy", 0.0)

        self.start_epoch = self.model.state.get("epoch", 0) + 1
        self.end_epoch = self.start_epoch + args.epochs

    def run(self):
        """Train the model."""
        log.started()

        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()
            train_stats = self.one_epoch(self.train_loader, self.optimizer)

            self.model.eval()
            val_stats = self.one_epoch(self.val_loader)

            is_best = self.save_checkpoint(val_stats, epoch)
            self.log_stats(train_stats, val_stats, epoch, is_best)

        log.finished()

    def one_epoch(self, loader, optimizer=None):
        """Train or validate an epoch."""
        running_loss = 0.0
        running_acc = 0.0

        for images, orders, targets, _ in loader:
            images = images.to(self.device)
            orders = orders.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(images, orders)
            loss = self.criterion(preds, targets)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(preds, targets)

        return {"loss": running_loss / len(loader), "acc": running_acc / len(loader)}

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.model.state.get("optimizer_state"):
            optimizer.load_state_dict(self.model.state["optimizer_state"])
        return optimizer

    def configure_criteria(self):
        """Configure the criterion for model improvement."""
        pos_weight = self.train_dataset.pos_weight().to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion

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

    @staticmethod
    def log_stats(train_stats, val_stats, epoch, is_best):
        """Log results of the epoch."""
        logging.info(
            f"{epoch:2}: "
            f"Train: loss {train_stats['loss']:0.6f} acc {train_stats['acc']:0.6f}\t"
            f"Valid: loss {val_stats['loss']:0.6f} acc {val_stats['acc']:0.6f}"
            f"{' ++' if is_best else ''}"
        )

    def save_checkpoint(self, val_stats, epoch):
        """Save the model if it meets criteria for being the current best model."""
        if (val_stats["acc"], -val_stats["loss"]) >= (self.best_acc, -self.best_loss):
            self.best_acc = val_stats["acc"]
            self.best_loss = val_stats["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_loss": self.best_loss,
                    "accuracy": self.best_acc,
                },
                self.save_model,
            )
            return True
        return False


class HerbariumTestRunner(HerbariumRunner):
    """Test the model."""

    def __init__(self, model, orders, args: ArgsType):
        super().__init__(model, orders, args)

        db.create_tests_table(args.database)

        self.split_set = args.split_set
        self.test_set = args.test_set
        self.target_set = args.target_set

        self.test_dataset = self.get_dataset(
            database=self.database,
            split_set=self.split_set,
            split="test",
            target_set=self.target_set,
            trait=self.trait,
            limit=self.limit,
        )

        self.test_loader = self.test_dataloader()
        self.criterion = self.configure_criteria()

    def test_dataloader(self):
        """Load the validation split for the data."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.workers
        )

    def configure_criteria(self):
        """Configure the criterion for model improvement."""
        pos_weight = self.test_dataset.pos_weight().to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion

    def run(self):
        """Test the model on hold-out data."""
        log.started()

        self.model.eval()

        test_loss = 0.0
        test_acc = 0.0

        batch = []

        for images, orders, targets, coreids in self.test_loader:
            images = images.to(self.device)
            orders = orders.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(images, orders)
            loss = self.criterion(preds, targets)

            test_loss += loss.item()
            test_acc += accuracy(preds, targets)

            preds = torch.sigmoid(preds)

            preds = preds.detach().cpu()
            targets = targets.detach().cpu()

            for target, pred, coreid in zip(targets, preds, coreids):
                batch.append(
                    {
                        "coreid": coreid,
                        "test_set": self.test_set,
                        "split_set": self.split_set,
                        "trait": self.trait,
                        "target": target.item(),
                        "pred": pred.item(),
                    }
                )

        db.insert_tests(self.database, batch, self.test_set, self.split_set)

        test_loss /= len(self.test_loader)
        test_acc /= len(self.test_loader)

        logging.info(f"Test: loss {test_loss:0.6f} acc {test_acc:0.6f}")
        log.finished()


class HerbariumInferenceRunner(HerbariumRunner):
    """Run inference on the model."""

    def __init__(self, model, orders, args: ArgsType):
        super().__init__(model, orders, args)

        self.inference_set = args.inference_set

        db.create_inferences_table(args.database)

        infer_recs = db.select_images(args.database, limit=args.limit)

        self.infer_dataset = InferenceDataset(infer_recs, model, orders=self.orders)
        self.infer_loader = self.infer_dataloader()

    def infer_dataloader(self):
        """Load the validation split for the data."""
        return DataLoader(
            self.infer_dataset, batch_size=self.batch_size, num_workers=self.workers
        )

    def run(self):
        """Run inference on images."""
        log.started()

        self.model.eval()

        batch = []

        for images, orders, _, coreids in self.infer_loader:
            images = images.to(self.device)
            orders = orders.to(self.device)

            preds = self.model(images, orders)
            preds = torch.sigmoid(preds)
            preds = preds.detach().cpu()

            for pred, coreid in zip(preds, coreids):
                batch.append(
                    {
                        "coreid": coreid,
                        "inference_set": self.inference_set,
                        "trait": self.trait,
                        "pred": pred.item(),
                    }
                )

        db.insert_inferences(self.database, batch, self.inference_set)

        log.finished()


def accuracy(preds, targets):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(preds))
    equals = (pred == targets).type(torch.float)
    return torch.mean(equals)
