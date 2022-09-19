"""Run inference on images."""
import logging
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import db_old
from ..datasets.unlabeled_dataset import UnlabeledDataset


def infer(model, orders, args: Namespace):
    """Run inference on images."""
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
    """Load the test data loader."""
    logging.info("Loading inference data.")
    db_old.create_inferences_table(args.database)
    raw_data = db_old.select_images(args.database, limit=args.limit)
    dataset = UnlabeledDataset(raw_data, model, orders=orders)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )


def insert_inference_records(database, batch, inference_set, trait):
    """Add inference records to the database."""
    db_old.create_inferences_table(database)

    for row in batch:
        row["inference_set"] = inference_set
        row["trait"] = trait

    db_old.insert_inferences(database, batch, inference_set)
