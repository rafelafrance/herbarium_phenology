"""Run inference on images."""
import logging
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import db
from ..datasets.unlabeled_dataset import UnlabeledDataset


def infer(model, orders, args: Namespace):
    device = torch.device("cuda" if torch.has_cuda else "cpu")

    infer_loader = get_data_loader(args, model, orders)

    batch = run_inference(model, device, infer_loader)
    insert_inference_records(args.database, batch, args.inference_set, args.trait)


def run_inference(model, device, loader):
    """Run inference for one epoch."""
    logging.info("Inference started.")

    model.eval()

    batch = []

    for images, orders, coreids in tqdm(loader):
        images = images.to(device)
        orders = orders.to(device)

        preds = model(images, orders)
        preds = torch.sigmoid(preds)
        preds = preds.detach().cpu()

        for pred, coreid in zip(preds, coreids):
            batch.append({"coreid": coreid, "pred": pred.item()})

    return batch


def get_data_loader(args, model, orders):
    logging.info("Loading inference data.")
    db.create_table(args.database, "inferences")
    raw_data = db.canned_select(args.database, "images", limit=args.limit)
    dataset = UnlabeledDataset(raw_data, model, orders=orders)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )


def insert_inference_records(database, batch, inference_set, trait):
    """Add inference records to the database."""
    db.create_table(database, "inferences")

    for row in batch:
        row["inference_set"] = inference_set
        row["trait"] = trait

    db.canned_delete(database, "inferences", inference_set=inference_set)
    db.canned_insert(database, "inferences", batch)
