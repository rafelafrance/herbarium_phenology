"""A model to classify herbarium traits."""
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import db
from .herbarium_dataset import HerbariumDataset


def train(args, net, orders):
    """Train a model."""
    best_loss = net.state.get("best_loss", np.Inf)
    model = net.model

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    train_split = db.select_split(
        args.database, args.split_run, split="train", limit=args.limit
    )
    train_dataset = HerbariumDataset(train_split, net, orders=orders, augment=True)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )

    val_split = db.select_split(
        args.database,
        args.split_run,
        split="val",
        limit=args.limit,
    )
    val_dataset = HerbariumDataset(val_split, net, orders=orders)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )

    pos_weight = train_dataset.pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_acc = one_epoch(
            model, train_loader, device, criterion, optimizer
        )

        model.eval()
        val_loss, val_acc = one_epoch(model, val_loader, device, criterion)

        flag = ""
        if val_loss <= best_loss:
            flag = "*"
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "accuracy": val_acc,
                },
                args.save_model,
            )

        print(
            f"{epoch:2}: Train: loss {train_loss:0.6f} acc {train_acc:0.6f}\t"
            f"Valid: loss {val_loss:0.6f} acc {val_acc:0.6f} {flag}\n"
        )


def test(args, net, orders):
    """Test the model on a hold-out data split."""
    model = net.model

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    test_split = db.select_split(
        args.database, args.split_run, split="test", limit=args.limit
    )
    test_dataset = HerbariumDataset(test_split, net, orders=orders)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )

    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    test_loss, test_acc = one_epoch(model, test_loader, device, criterion)

    print(f"Test: loss {test_loss:0.6f} acc {test_acc:0.6f}")


def one_epoch(model, loader, device, criterion, optimizer=None):
    """Train an epoch."""
    total_loss = 0.0
    acc = 0.0

    for images, orders, y_true in tqdm(loader):
        images = images.to(device)
        orders = orders.to(device)
        y_true = y_true.to(device)

        y_pred = model(images, orders)
        loss = criterion(y_pred, y_true)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        acc += accuracy(y_pred, y_true)

    return total_loss / len(loader), acc / len(loader)


def accuracy(y_pred, y_true):
    """Calculate the accuracy of the model."""
    # pred = torch.round(y_pred)
    pred = torch.round(functional.softmax(y_pred, dim=1))
    equals = (pred == y_true).type(torch.float)
    return torch.mean(equals)
