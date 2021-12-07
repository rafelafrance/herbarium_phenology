"""A model to classify herbarium traits."""
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import db
from .herbarium_dataset import HerbariumDataset


class Classifier:
    """A class for training efficient net models."""

    def __init__(self, args):
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.save_model = args.save_model
        self.epochs = args.epochs

        self.model = self.get_model()

        state = torch.load(args.prev_model) if args.prev_model else {}
        if state.get("model_state"):
            self.model.load_state_dict(state["model_state"])

        self.best_loss = state.get("best_loss", np.Inf)


        self.device = torch.device("cuda" if torch.has_cuda else "cpu")
        self.model.to(self.device)

        train_data = db.select_split(args.database, args.split_run, "train")
        train_dataset = HerbariumDataset(train_data, augment=True)
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
        )

        val_data = db.select_split(args.database, args.split_run, "val")
        val_dataset = HerbariumDataset(val_data)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
        )

        pos_weight = train_dataset.pos_weight().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    @staticmethod
    def get_model():
        """Get the model to use."""
        raise NotImplemented()

    def train(self):
        """Train the model."""
        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_acc = self.train_epoch()

            self.model.eval()
            val_loss, val_acc = self.val_epoch()

            flag = ""
            if val_loss <= self.best_loss:
                flag = "*"
                self.best_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_accuracy": val_acc,
                    "best_loss": self.best_loss,
                    },
                    self.save_model)

            print(f"{epoch:2}: Train: loss {train_loss:0.6f} acc {train_acc:0.6f}\t"
                  f"Valid: loss {val_loss:0.6f} acc {val_acc:0.6f} {flag}\n")

    def train_epoch(self):
        """Train an epoch."""
        total_loss = 0.0
        accuracy = 0.0

        for images, y_true in tqdm(self.train_loader):
            images = images.to(self.device)
            y_true = y_true.to(self.device)

            y_pred = self.model(images)

            loss = self.criterion(y_pred, y_true)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            accuracy += self.accuracy(y_pred, y_true)

        return total_loss / len(self.train_loader), accuracy / len(self.train_loader)

    def val_epoch(self):
        """Train an epoch."""
        total_loss = 0.0
        accuracy = 0.0

        for images, y_true in tqdm(self.val_loader):
            images = images.to(self.device)
            y_true = y_true.to(self.device)

            y_pred = self.model(images)
            loss = self.criterion(y_pred, y_true)

            total_loss += loss.item()
            accuracy += self.accuracy(y_pred, y_true)

        return total_loss / len(self.val_loader), accuracy / len(self.val_loader)

    @staticmethod
    def accuracy(y_pred, y_true):
        """Calculate the accuracy of the model."""
        pred = torch.round(functional.softmax(y_pred, dim=1))
        equals = (pred == y_true).type(torch.float)
        return torch.mean(equals)


class EfficientNetB0(Classifier):
    """A class for training efficient net models."""

    @staticmethod
    def get_model():
        """Get the model to use."""
        model = torchvision.models.efficientnet_b0(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=1280),
            nn.Linear(in_features=1280, out_features=480),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=480),
            nn.Linear(in_features=480, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=len(HerbariumDataset.all_classes)),
        )
        return model


class EfficientNetB4(Classifier):
    """A class for training efficient net models."""

    @staticmethod
    def get_model():
        """Get the model to use."""
        model = torchvision.models.efficientnet_b4(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=1792),
            nn.Linear(in_features=1792, out_features=625),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=625),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=len(HerbariumDataset.all_classes)),
        )
        return model
