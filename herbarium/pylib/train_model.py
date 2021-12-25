"""A model to classify herbarium traits."""
import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from . import db
from . import log
from .herbarium_dataset import HerbariumDataset


def train(args, model, orders):
    """Train a model."""
    log.started()

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    train_split = db.select_split(
        args.database, args.split_run, split="train", limit=args.limit
    )
    train_dataset = HerbariumDataset(
        train_split, model, orders=orders, traits=args.trait, augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=len(train_split) % args.batch_size == 1,
    )

    val_split = db.select_split(
        args.database,
        args.split_run,
        split="val",
        limit=args.limit,
    )
    val_dataset = HerbariumDataset(val_split, model, orders=orders, traits=args.trait)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=len(val_split) % args.batch_size == 1,
    )

    pos_weight = train_dataset.pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = load_optimizer(model, args.learning_rate, device)

    best_loss = model.state.get("best_loss", np.Inf)
    best_acc = model.state.get("accuracy", 0.0)

    start_epoch = model.state.get("epoch", 0) + 1
    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_loss, train_acc = one_epoch(
            model, train_loader, device, criterion, optimizer
        )

        model.eval()
        val_loss, val_acc = one_epoch(model, val_loader, device, criterion)

        flag = ""
        if val_loss <= best_loss:
            best_loss = val_loss
            file_name = args.save_model.with_stem(args.save_model.stem + "_loss")
            flag += " --"
            save_model(model, optimizer, epoch, best_loss, val_acc, file_name)

        if val_acc >= best_acc:
            best_acc = val_acc
            file_name = args.save_model.with_stem(args.save_model.stem + "_acc")
            flag += " ++"
            save_model(model, optimizer, epoch, best_loss, best_acc, file_name)

        if epoch % 10 == 0:
            save_model(model, optimizer, epoch, best_loss, best_acc, args.save_model)

        logging.info(
            f"{epoch:2}: Train: loss {train_loss:0.6f} acc {train_acc:0.6f}\t"
            f"Valid: loss {val_loss:0.6f} acc {val_acc:0.6f}{flag}"
        )

    save_model(model, optimizer, end_epoch - 1, best_loss, best_acc, args.save_model)
    log.finished()


def test(args, model, orders):
    """Test the model on a hold-out data split."""
    log.started()

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    test_split = db.select_split(
        args.database, args.split_run, split="test", limit=args.limit
    )
    test_dataset = HerbariumDataset(test_split, model, orders=orders, traits=args.trait)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=len(test_split) % args.batch_size == 1,
    )

    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    test_loss, test_acc = one_epoch(model, test_loader, device, criterion)

    logging.info(f"Test: loss {test_loss:0.6f} acc {test_acc:0.6f}")
    log.finished()


def one_epoch(model, loader, device, criterion, optimizer=None):
    """Train an epoch."""
    avg_loss = 0.0
    avg_acc = 0.0
    # torch.autograd.set_detect_anomaly(True)

    for images, orders, y_true in loader:
        images = images.to(device)
        orders = orders.to(device)
        y_true = y_true.to(device)

        y_pred = model(images, orders)
        loss = criterion(y_pred, y_true)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss += loss.item()
        avg_acc += accuracy(y_pred, y_true)

    return avg_loss / len(loader), avg_acc / len(loader)


def load_optimizer(model, learning_rate, device):
    """Create an optimizer."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if model.state.get("optimizer_state"):
        optimizer.load_state_dict(model.state["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    return optimizer


def save_model(model, optimizer, epoch, best_loss, best_acc, file_name):
    """Save the model to disk."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "accuracy": best_acc,
        },
        file_name,
    )


def accuracy(y_pred, y_true):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(y_pred))
    equals = (pred == y_true).type(torch.float)
    return torch.mean(equals)
