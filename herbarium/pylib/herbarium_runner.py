"""Run a hydra model for training, testing, or inference."""
import logging
import random
from abc import ABC
from abc import abstractmethod
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import db
from . import log
from .herbarium_dataset import HerbariumDataset
from .herbarium_dataset import InferenceDataset
from .herbarium_dataset import PseudoDataset

ArgsType = Namespace


class HerbariumRunner(ABC):
    """Base class for running a herbarium model."""

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

    @abstractmethod
    def run(self):
        """Run the function of the class"""


class HerbariumTrainingRunner(HerbariumRunner):
    """Train a herbarium model."""

    def __init__(self, model, orders, args: ArgsType):
        super().__init__(model, orders, args)

        self.lr = args.learning_rate
        self.split_set = args.split_set
        self.save_model = args.save_model
        self.target_set = args.target_set

        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer = self.configure_optimizers()
        self.criterion = self.configure_criterion(self.train_loader.dataset)

        self.best_loss = self.model.state.get("best_loss", np.Inf)
        self.best_acc = self.model.state.get("accuracy", 0.0)
        self.run_loss = np.Inf
        self.run_acc = 0.0

        self.start_epoch = self.model.state.get("epoch", 0) + 1
        self.end_epoch = self.start_epoch + args.epochs

    def run(self):
        """Train the model."""
        log.started()

        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()
            train_stats = self.one_epoch(
                self.train_loader, self.criterion, self.optimizer
            )

            self.model.eval()
            val_stats = self.one_epoch(self.val_loader, self.criterion)

            is_best = self.save_checkpoint(val_stats, epoch)
            self.log_stats(train_stats, val_stats, epoch, is_best)

        log.finished()

    def one_epoch(self, loader, criterion, optimizer=None):
        """Train or validate an epoch."""
        running_loss = 0.0
        running_acc = 0.0

        for images, orders, targets, _ in loader:
            images = images.to(self.device)
            orders = orders.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(images, orders)
            loss = criterion(preds, targets)

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

    def configure_criterion(self, dataset):
        """Configure the criterion for model improvement."""
        pos_weight = dataset.pos_weight()
        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion

    def train_dataloader(self):
        """Load the training split for the data."""
        raw_data = db.select_split(
            database=self.database,
            split_set=self.split_set,
            split="train",
            target_set=self.target_set,
            trait=self.trait,
            limit=self.limit,
        )
        dataset = HerbariumDataset(
            raw_data,
            self.model,
            orders=self.orders,
            augment=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            pin_memory=True,
            drop_last=len(dataset) % self.batch_size == 1,
        )

    def val_dataloader(self):
        """Load the validation split for the data."""
        raw_data = db.select_split(
            database=self.database,
            split_set=self.split_set,
            split="val",
            target_set=self.target_set,
            trait=self.trait,
            limit=self.limit,
        )
        dataset = HerbariumDataset(
            raw_data,
            self.model,
            orders=self.orders,
            augment=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
        )

    @staticmethod
    def log_stats(train_stats, val_stats, epoch, is_best):
        """Log results of the epoch."""
        logging.info(
            f"{epoch:2}: "
            f"Train: loss {train_stats['loss']:0.6f} acc {train_stats['acc']:0.6f} "
            f"Valid: loss {val_stats['loss']:0.6f} acc {val_stats['acc']:0.6f}"
            f"{' ++' if is_best else ''}"
        )

    def save_checkpoint(self, val_stats, epoch):
        """Save the model if it meets criteria for being the current best model."""
        if (val_stats["acc"], -val_stats["loss"]) >= (self.run_acc, -self.run_loss):
            self.run_acc = val_stats["acc"]
            self.run_loss = val_stats["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_loss": self.best_loss,
                    "accuracy": self.best_acc,
                },
                self.save_model.with_stem(self.save_model.stem + "_chk"),
            )

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

        self.test_loader = self.test_dataloader()
        self.criterion = self.configure_criterion(self.test_loader.dataset)

    def test_dataloader(self):
        """Load the validation split for the data."""
        raw_data = db.select_split(
            database=self.database,
            split_set=self.split_set,
            split="test",
            target_set=self.target_set,
            trait=self.trait,
            limit=self.limit,
        )
        dataset = HerbariumDataset(
            raw_data,
            self.model,
            orders=self.orders,
            augment=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
        )

    def configure_criterion(self, dataset):
        """Configure the criterion for model improvement."""
        pos_weight = dataset.pos_weight()
        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(self.device)
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

        db.create_inferences_table(self.database)

        self.infer_loader = self.infer_dataloader()

    def infer_dataloader(self):
        """Load the validation split for the data."""
        raw_data = db.select_images(self.database, limit=self.limit)
        dataset = InferenceDataset(raw_data, self.model, orders=self.orders)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
        )

    def run(self):
        """Run inference on images."""
        log.started()

        self.model.eval()

        batch = []

        for images, orders, coreids in tqdm(self.infer_loader):
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


class HerbariumPseudoRunner(HerbariumTrainingRunner):
    """Train a herbarium model with pseudo-labels."""

    def __init__(self, model, orders, args: ArgsType):
        super().__init__(model, orders, args)

        self.inference_set = args.inference_set
        self.min_threshold = args.min_threshold
        self.max_threshold = args.max_threshold

        self.pseudo_loader = self.pseudo_dataloader()
        self.pseudo_criterion = self.configure_criterion(self.pseudo_loader.dataset)

    def run(self):
        """Run train the model using pseudo-labels."""
        log.started()

        pseudo_prob = 0.1
        pseudo_max = 0.9
        pseudo_step = 0.1
        pseudo_update = 10

        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()

            if epoch % pseudo_update == 0 and pseudo_prob < pseudo_max:
                pseudo_prob += pseudo_step

            if random.random() >= pseudo_prob:
                train_stats = self.one_epoch(
                    self.train_loader, self.criterion, self.optimizer
                )
                is_pseudo = False
            else:
                train_stats = self.one_epoch(
                    self.pseudo_loader, self.pseudo_criterion, self.optimizer
                )
                is_pseudo = True

            self.model.eval()
            val_stats = self.one_epoch(self.val_loader, self.criterion)

            is_best = self.save_checkpoint(val_stats, epoch)
            self.logger(train_stats, val_stats, epoch, is_best, is_pseudo)

        log.finished()

    @staticmethod
    def logger(train_stats, val_stats, epoch, is_best, is_pseudo):
        """Log results of the epoch."""
        logging.info(
            f"{epoch:2}: "
            f"Train: loss {train_stats['loss']:0.6f} acc {train_stats['acc']:0.6f} "
            f"Valid: loss {val_stats['loss']:0.6f} acc {val_stats['acc']:0.6f}"
            f" {'p' if is_pseudo else ' '}"
            f"{' ++' if is_best else ''}"
        )

    def pseudo_dataloader(self):
        """Load the pseudo-dataset."""
        raw_data = db.select_pseudo_split(
            database=self.database,
            trait=self.trait,
            inference_set=self.inference_set,
            target_set=self.target_set,
            min_threshold=self.min_threshold,
            max_threshold=self.max_threshold,
            limit=self.limit,
        )
        dataset = PseudoDataset(
            raw_data,
            self.model,
            orders=self.orders,
            min_threshold=self.min_threshold,
            max_threshold=self.max_threshold,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            pin_memory=True,
            drop_last=len(dataset) % self.batch_size == 1,
        )


def accuracy(preds, targets):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(preds))
    equals = (pred == targets).type(torch.float)
    return torch.mean(equals)
