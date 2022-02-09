"""Run a model for training, testing, or inference."""
import logging

import torch
from torch.utils.data import DataLoader

from . import db
from .herbarium_dataset import InferenceDataset
from .herbarium_runner import ArgsType
from .herbarium_runner import HerbariumTrainingRunner


class HerbariumPseudoRunner(HerbariumTrainingRunner):
    """Train a herbarium model with pseudo-labels."""

    def __init__(self, model, orders, args: ArgsType):
        super().__init__(model, orders, args)

        self.pseudo_max = args.pseudo_max
        self.pseudo_start = args.pseudo_start
        self.pseudo_loader = self.pseudo_dataloader(args.unlabeled_limit)

    def run(self):
        """Run train the model using pseudo-labels."""
        logging.info("Pseudo-training started.")

        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()
            train_stats = self.one_epoch(
                self.train_loader, self.loss_fn, self.optimizer
            )

            alpha = self.alpha(epoch)
            pseudo_stats = {"loss": 0.0}
            if alpha > 0.0:
                pseudo_stats = self.pseudo_epoch(
                    self.pseudo_loader, self.loss_fn, self.optimizer, alpha
                )

            self.model.eval()
            val_stats = self.one_epoch(self.val_loader, self.loss_fn)

            is_best = self.save_checkpoint(val_stats, epoch)
            self.logger(train_stats, pseudo_stats, val_stats, epoch, is_best, alpha)

        self.writer.close()

    def pseudo_epoch(self, loader, loss_fn, optimizer, alpha):
        """Train or validate an epoch."""
        running_loss = 0.0

        for images, orders, _ in loader:
            images = images.to(self.device)
            orders = orders.to(self.device)

            preds = self.model(images, orders)
            targets = torch.round(torch.sigmoid(preds))

            loss = alpha * loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return {"loss": running_loss / len(loader)}

    def alpha(self, epoch):
        """Calculate the loss weight for the pseudo labels."""
        e = max(0, epoch - self.pseudo_start)
        return round(min(epoch / 100, self.pseudo_max), 2) if e > 0 else 0.0

    def pseudo_dataloader(self, unlabeled_limit):
        """Load the pseudo-dataset."""
        logging.info("Loading pseudo-training data.")
        raw_data = db.select_pseudo_split(
            database=self.database,
            target_set=self.target_set,
            trait=self.trait,
            limit=unlabeled_limit,
        )
        dataset = InferenceDataset(raw_data, self.model, orders=self.orders)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            pin_memory=True,
            drop_last=len(dataset) % self.batch_size == 1,
        )

    def logger(self, train_stats, pseudo_stats, val_stats, epoch, is_best, alpha):
        """Log results of the epoch."""
        logging.info(
            f"{epoch:4}: "
            f"Train: loss {train_stats['loss']:0.6f} acc {train_stats['acc']:0.6f} "
            f"Pseudo: loss {pseudo_stats['loss']:0.6f} "
            f"Valid: loss {val_stats['loss']:0.6f} acc {val_stats['acc']:0.6f}"
            f"{' ++' if is_best else ''}"
        )
        self.writer.add_scalars(
            "Training/Pseudo vs. Validation",
            {
                "Training loss": train_stats["loss"],
                "Training accuracy": train_stats["acc"],
                "Pseudo alpha": alpha,
                "Pseudo loss": pseudo_stats["loss"],
                "Validation loss": val_stats["loss"],
                "Validation accuracy": val_stats["acc"],
            },
            epoch,
        )
        self.writer.flush()
