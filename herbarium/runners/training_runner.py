"""Train a model to classify herbarium traits."""
import argparse
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from herbarium.datasets.herbarium_dataset import HerbariumDataset
from herbarium.pylib import db
from herbarium.runners import runner_utils


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


def train(model, orders, args: argparse.Namespace):
    """Train a herbarium model."""
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    # use_col = const.TRAIT_2_INT[args.trait]  # for hydra

    train_loader = get_train_loader(args, model, orders)
    val_loader = get_val_loader(args, model, orders)

    optimizer = get_optimizer(model, args.learning_rate)
    scheduler = get_scheduler(optimizer)
    loss_fn = get_loss_fn(train_loader.dataset, device)

    writer = SummaryWriter(args.log_dir)

    stats = Stats(
        best_acc=model.state.get("accuracy", 0.0),
        best_loss=model.state.get("best_loss", np.Inf),
    )

    end_epoch, start_epoch = get_epoch_range(args, model)

    logging.info("Training started.")
    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_acc, stats.train_loss = one_epoch(
            model, device, train_loader, loss_fn, optimizer
        )

        if scheduler:
            scheduler.step()

        model.eval()
        stats.val_acc, stats.val_loss = one_epoch(model, device, val_loader, loss_fn)

        save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(writer, stats, epoch)

    writer.close()


def one_epoch(model, device, loader, loss_fn, optimizer=None):
    """Train or validate an epoch."""
    running_loss = 0.0
    running_acc = 0.0

    for images, orders, targets, _ in loader:
        images = images.to(device)
        orders = orders.to(device)
        targets = targets.to(device)

        preds = model(images, orders)
        # preds = preds[:, use_col].to(device)  # for hydra

        loss = loss_fn(preds, targets)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += runner_utils.accuracy(preds, targets)

    return running_acc / len(loader), running_loss / len(loader)


def get_epoch_range(args, model):
    """Calculate the epoch range."""
    start_epoch = model.state.get("epoch", 0) + 1
    end_epoch = start_epoch + args.epochs
    return end_epoch, start_epoch


def get_train_loader(args, model, orders):
    """Build the training data loader."""
    logging.info("Loading training data.")
    raw_data = db.select_split(
        database=args.database,
        split_set=args.split_set,
        split="train",
        target_set=args.target_set,
        trait=args.trait,
        limit=args.limit,
    )
    dataset = HerbariumDataset(raw_data, model, orders=orders, augment=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=len(dataset) % args.batch_size == 1,
    )


def get_val_loader(args, model, orders):
    """Build the validation data loader."""
    logging.info("Loading validation data.")
    raw_data = db.select_split(
        database=args.database,
        split_set=args.split_set,
        split="val",
        target_set=args.target_set,
        trait=args.trait,
        limit=args.limit,
    )
    dataset = HerbariumDataset(raw_data, model, orders=orders, augment=False)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )


def get_optimizer(model, lr):
    """Configure the optimizer."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if model.state.get("optimizer_state"):
        logging.info("Loading the optimizer.")
        optimizer.load_state_dict(model.state["optimizer_state"])
    return optimizer


def get_scheduler(optimizer):
    """Schedule the rate of change in the optimizer."""
    return optim.lr_scheduler.CyclicLR(
        optimizer, 0.001, 0.01, step_size_up=10, cycle_momentum=False
    )


def get_loss_fn(dataset, device):
    """Configure the loss_fn for model improvement."""
    pos_weight = dataset.pos_weight()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn


def save_checkpoint(model, optimizer, save_model, stats, epoch):
    """Save the model if it meets criteria for being the current best model."""
    stats.is_best = False
    if (stats.val_acc, -stats.val_loss) >= (stats.best_acc, -stats.best_loss):
        stats.is_best = True
        stats.best_acc = stats.val_acc
        stats.best_loss = stats.val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": stats.best_loss,
                "accuracy": stats.best_acc,
            },
            save_model,
        )


def log_stats(writer, stats, epoch):
    """Log results of the epoch."""
    logging.info(
        f"{epoch:4}: "
        f"Train: loss {stats.train_loss:0.6f} acc {stats.train_acc:0.6f} "
        f"Valid: loss {stats.val_loss:0.6f} acc {stats.val_acc:0.6f}"
        f"{' ++' if stats.is_best else ''}"
    )
    writer.add_scalars(
        "Training vs. Validation",
        {
            "Training loss": stats.train_loss,
            "Training accuracy": stats.train_acc,
            "Validation loss": stats.val_loss,
            "Validation accuracy": stats.val_acc,
        },
        epoch,
    )
    writer.flush()
