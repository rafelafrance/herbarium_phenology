"""Model architectures used."""
from typing import Optional

import torch
import torchvision
from torch import nn
from torchvision.models import EfficientNet

from .herbarium_dataset import HerbariumDataset


class BaseNet:
    """Base class for all the EfficientNet classes."""

    def __init__(self, args):
        self.model: Optional[EfficientNet] = None
        self.freeze = args.freeze
        self.state = torch.load(args.prev_model) if args.prev_model else {}

    def freeze_all(self):
        """Freeze the layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Freeze the layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = True


class EfficientNetB0(BaseNet):
    """A class for training efficient net models."""

    def __init__(self, args):
        super().__init__(args)

        self.size = (224, 224)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2250, 0.2326, 0.2449]

        self.model = torchvision.models.efficientnet_b0(pretrained=True)

        if self.freeze:
            self.freeze_all()

        self.model.classifier = nn.Sequential(
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

        if self.state.get("model_state"):
            self.model.load_state_dict(self.state["model_state"])


class EfficientNetB4(BaseNet):
    """A class for training efficient net models."""

    def __init__(self, args):
        super().__init__(args)

        self.size = (380, 380)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2286, 0.2365, 0.2492]

        self.model = torchvision.models.efficientnet_b4(pretrained=True)

        if self.freeze:
            self.freeze_all()

        self.model.classifier = nn.Sequential(
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

        if self.state.get("model_state"):
            self.model.load_state_dict(self.state["model_state"])


NETS = {
    "b0": EfficientNetB0,
    "b4": EfficientNetB4,
}
