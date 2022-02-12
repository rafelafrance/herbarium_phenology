"""Try adding the phylogenetic orders at several classifier layers."""
from pathlib import Path

import torch
from torch import nn

from . import model_utils
from .backbones import BACKBONES
from .base_model import BaseBackbone


class EchoHead(nn.Module):
    """Classify a trait using backbone output & phylogenetic orders as sidecar data."""

    def __init__(self, orders: list[str], backbone: str):
        super().__init__()

        self.orders = orders

        model_params = BACKBONES[backbone]

        fc1_in = model_params["in_feat"] + len(orders)
        fc1_out = fc1_in // 4
        fc2_in = fc1_out + len(orders)
        fc2_out = fc2_in // 4
        fc3_in = fc2_out + len(orders)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=fc1_in, out_features=fc1_out, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc1_out),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=fc2_in, out_features=fc2_out, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc2_out),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=fc3_in, out_features=1),
        )

    def forward(self, x0, x1):
        """Run the classifier forwards with a phylogenetic order (one-hot)."""
        x = torch.cat((x0, x1), dim=1)
        x = self.fc1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.fc2(x)
        x = torch.cat((x, x1), dim=1)
        x = self.fc3(x)
        return x


class EchoModel(nn.Module):
    """The utils model for a single trait."""

    def __init__(
        self, orders: list[str], backbone: str, load_model: Path, trait: str = None
    ):
        super().__init__()

        self.trait = trait

        model_utils.get_backbone_params(self, backbone)

        self.backbone = BaseBackbone(backbone)
        self.head = EchoHead(orders, backbone)

        model_utils.load_model_state(self, load_model)

    def forward(self, x0, x1):
        """Feed the backbone to all of the classifiers."""
        x0 = self.backbone(x0)
        x = self.head(x0, x1)
        return x
