import logging
from argparse import Namespace
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import db
from ..datasets.labeled_dataset import LabeledDataset
from ..engines import engine_utils


@dataclass
class Stats:
    acc: float
    loss: float


def test(model, orders, args: Namespace):
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    # use_col = const.TRAIT_2_INT[args.trait]  # for hydra

    test_loader = get_data_loader(args, model, orders)
    loss_fn = get_loss_fn(test_loader.dataset, device)

    batch, stats = run_test(model, device, test_loader, loss_fn)

    insert_test_records(args.database, batch, args.test_set, args.split_set, args.trait)

    logging.info(f"Test: loss {stats.loss:0.6f} acc {stats.acc:0.6f}")


def run_test(model, device, loader, loss_fn):
    logging.info("Testing started.")

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    batch = []

    for images, orders, targets, coreids in tqdm(loader):
        images = images.to(device)
        orders = orders.to(device)
        targets = targets.to(device)

        preds = model(images, orders)
        loss = loss_fn(preds, targets)

        running_loss += loss.item()
        running_acc += engine_utils.accuracy(preds, targets)

        preds = torch.sigmoid(preds)

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        for target, pred, coreid in zip(targets, preds, coreids):
            batch.append(
                {
                    "coreid": coreid,
                    "target": target.item(),
                    "pred": pred.item(),
                }
            )

    return batch, Stats(acc=running_acc / len(loader), loss=running_loss / len(loader))


def insert_test_records(database, batch, test_set, split_set, trait):
    """Add test records to the database."""
    db.create_table(database, "tests")

    for row in batch:
        row["test_set"] = test_set
        row["split_set"] = split_set
        row["trait"] = trait

    db.canned_delete(database, "tests", test_set=test_set, split_set=split_set)
    db.canned_insert(database, batch, test_set)


def get_loss_fn(dataset, device):
    """Configure the loss_fn for model improvement."""
    pos_weight = dataset.pos_weight()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn


def get_data_loader(args, model, orders):
    """Build the test data loader."""
    logging.info("Loading test data.")
    raw_data = db.canned_select(
        args.database,
        "split",
        split_set=args.split_set,
        split="test",
        target_set=args.target_set,
        trait=args.trait,
        limit=args.limit,
    )
    dataset = LabeledDataset(raw_data, model, orders=orders, augment=False)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
