"""Run a model for classification in training, testing, & inference modes."""
import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from . import db
from . import log
from .herbarium_hydra_dataset import HerbariumHydraDataset

# from torch.utils.tensorboard import SummaryWriter


def train(args, model, orders):
    """Train a model."""
    log.started()
    # writer = SummaryWriter()

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    train_split = db.select_split(
        args.database, args.split_run, split="train", limit=args.limit
    )
    train_dataset = HerbariumHydraDataset(
        train_split, model, orders=orders, augment=True
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
    val_dataset = HerbariumHydraDataset(val_split, model, orders=orders)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.workers
    )

    pos_weight = train_dataset.pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = load_optimizer(model, args.learning_rate)

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
        if (val_acc, -val_loss) >= (best_acc, -best_loss):
            best_acc = val_acc
            file_name = args.save_model.with_stem(args.save_model.stem + "_acc")
            flag += " ++"
            save_model(model, optimizer, epoch, best_loss, best_acc, file_name)

        logging.info(
            f"{epoch:2}: Train: loss {train_loss:0.6f} acc {train_acc:0.6f}\t"
            f"Valid: loss {val_loss:0.6f} acc {val_acc:0.6f}{flag}"
        )

    log.finished()


def test(args, model, orders):
    """Test the model on a hold-out/test data split."""
    log.started()

    db.create_test_runs_table(args.database)

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    test_split = db.select_split(
        args.database, args.split_run, split="test", limit=args.limit
    )
    test_dataset = HerbariumHydraDataset(test_split, model, orders=orders)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    test_loss = 0.0
    test_acc = 0.0

    batch = []

    for images, orders, y_true, coreids in test_loader:
        images = images.to(device)
        orders = orders.to(device)
        y_true = y_true.to(device)

        y_pred = model(images, orders)
        loss = criterion(y_pred, y_true)

        test_loss += loss.item()
        test_acc += accuracy(y_pred, y_true)

        y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()

        for trues, preds, coreid in zip(y_true, y_pred, coreids):
            for trait, true, pred in zip(args.trait, trues, preds):
                batch.append(
                    {
                        "coreid": coreid,
                        "test_run": args.test_run,
                        "split_run": args.split_run,
                        "trait": trait,
                        "true": true.item(),
                        "pred": pred.item(),
                    }
                )

    db.insert_test_runs(args.database, batch, args.test_run, args.split_run)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    logging.info(f"Test: loss {test_loss:0.6f} acc {test_acc:0.6f}")
    log.finished()


def infer(args, model, orders):
    """Test the model on a hold-out/test data split."""
    log.started()

    db.create_inferences_table(args.database)

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    infer_split = db.select_images(args.database, limit=args.limit)
    infer_dataset = HerbariumHydraDataset(infer_split, model, orders=orders)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model.eval()

    batch = []

    for images, orders, _, coreids in infer_loader:
        images = images.to(device)
        orders = orders.to(device)

        y_pred = model(images, orders)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.detach().cpu()

        for preds, coreid in zip(y_pred, coreids):
            for trait, pred in zip(args.trait, preds):
                batch.append(
                    {
                        "coreid": coreid,
                        "inference_run": args.inference_run,
                        "trait": trait,
                        "pred": pred.item(),
                    }
                )

    db.insert_inferences(args.database, batch, args.inference_run)

    log.finished()


def one_epoch(model, loader, device, criterion, optimizer=None):
    """Train or validate an epoch."""
    avg_loss = 0.0
    avg_acc = 0.0

    for images, orders, y_true, _ in loader:
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


def load_optimizer(model, learning_rate):
    """Create an optimizer."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if model.state.get("optimizer_state"):
        optimizer.load_state_dict(model.state["optimizer_state"])
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
