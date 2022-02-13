"""Run a model for training, testing, or inference."""
import argparse
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import training_runner as tr
from .. import db
from ..datasets.unlabeled_dataset import UnlabeledDataset


@dataclass
class Stats:
    """Gather statistics while training."""

    is_best: bool = False
    best_acc: float = 0.0
    best_loss: float = np.Inf
    train_acc: float = 0.0
    train_loss: float = np.Inf
    val_acc: float = 0.0
    val_loss: float = np.Inf
    pseudo_loss: float = np.Inf


def train(model, orders, args: argparse.Namespace):
    """Train a utils model with pseudo-labels."""
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    pseudo_loader = get_pseudo_loader(args, model, orders)
    train_loader = tr.get_train_loader(args, model, orders)
    val_loader = tr.get_val_loader(args, model, orders)

    optimizer = tr.get_optimizer(model, args.learning_rate)
    loss_fn = tr.get_loss_fn(train_loader.dataset, device)

    writer = SummaryWriter(args.log_dir)

    stats = Stats(
        best_acc=model.state.get("accuracy", 0.0),
        best_loss=model.state.get("best_loss", np.Inf),
    )

    logging.info("Pseudo-training started.")
    end_epoch, start_epoch = tr.get_epoch_range(args, model)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_acc, stats.train_loss = tr.one_epoch(
            model, device, train_loader, loss_fn, optimizer
        )

        alpha_value = alpha(epoch, args.pseudo_start, args.pseudo_max)
        if alpha_value > 0.0:
            stats.pseudo_loss = pseudo_epoch(
                model, device, pseudo_loader, loss_fn, optimizer, alpha_value
            )

        model.eval()
        stats.val_acc, stats.val_loss = tr.one_epoch(model, device, val_loader, loss_fn)

        tr.save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(writer, stats, epoch, alpha_value)

    writer.close()


def pseudo_epoch(model, device, loader, loss_fn, optimizer, alpha_value):
    """Train or validate an epoch."""
    running_loss = 0.0

    for images, orders, _ in loader:
        images = images.to(device)
        orders = orders.to(device)

        preds = model(images, orders)
        targets = torch.round(torch.sigmoid(preds))

        loss = alpha_value * loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def alpha(epoch, pseudo_start, pseudo_max):
    """Calculate the loss weight for the pseudo labels."""
    e = max(0, epoch - pseudo_start)
    return round(min(epoch / 100, pseudo_max), 2) if e > 0 else 0.0


def get_pseudo_loader(args, model, orders):
    """Load the pseudo-dataset and loader."""
    logging.info("Loading pseudo-training data.")
    raw_data = db.select_pseudo_split(
        database=args.database,
        target_set=args.target_set,
        trait=args.trait,
        limit=args.unlabeled_limit,
    )
    dataset = UnlabeledDataset(raw_data, model, orders=orders)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=len(dataset) % args.batch_size == 1,
    )


def log_stats(writer, stats, epoch, alpha_value):
    """Log results of the epoch."""
    logging.info(
        f"{epoch:4}: "
        f"Train: loss {stats.train_loss:0.6f} acc {stats.train_acc:0.6f} "
        f"Pseudo: loss {stats.pseudo_loss:0.6f} "
        f"Valid: loss {stats.val_loss:0.6f} acc {stats.val_acc:0.6f}"
        f"{' ++' if stats.is_best else ''}"
    )
    writer.add_scalars(
        "Training vs. Pseudo vs. Validation",
        {
            "Training loss": stats.train_loss,
            "Training accuracy": stats.train_acc,
            "Pseudo alpha": alpha_value,
            "Pseudo loss": stats.pseudo_loss,
            "Validation loss": stats.val_loss,
            "Validation accuracy": stats.val_acc,
        },
        epoch,
    )
    writer.flush()
