"""Train a model to classify herbarium traits."""
import sqlite3

import torch
import torchvision
from torch import nn
from torch import optim

from pylib import db
from pylib import util
from pylib.herbarium_dataset import HerbariumDataset


def train(args):
    """Train the model."""
    torch.multiprocessing.set_sharing_strategy("file_system")

    state = torch.load(args.prev_model) if args.prev_model else {}

    model = get_model()
    if state.get("model_state"):
        model.load_state_dict(state["model_state"])

    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset, val_dataset = get_datasets(
        args.database, args.size, args.mean, args.std_dev, args.split
    )


def get_datasets(database, image_size, mean, std_dev, split):
    """Get the training and validation datasets."""
    records = {}
    for cls in HerbariumDataset.all_classes:
        sql = f"select * from angiosperms join images using (coreid) where {cls} = 1"
        records[cls] = db.rows_as_dicts(database, sql)
    # train_dataset, val_dataset = records[:split], records[split:]
    # return train_dataset, val_dataset
    return [], []


def get_model():
    """Get the model to use."""
    model = torchvision.models.efficientnet_b4(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1792, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=len(HerbariumDataset.all_classes)),
    )
    return model
