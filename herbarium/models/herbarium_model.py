"""Create EfficientNets that use plant orders as well as images for input."""
import logging
from pathlib import Path

import torch
from torch import nn

from .backbone_params import BACKBONES
from .backbone_params import IMAGENET_MEAN
from .backbone_params import IMAGENET_STD_DEV


class HerbariumBackbone(nn.Module):
    """Pretrained backbone."""

    def __init__(self, backbone: str):
        super().__init__()

        model_params = BACKBONES[backbone]

        self.model = model_params["backbone"](pretrained=True)
        self.model.classifier = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Build image output."""
        return self.model(x)


class HerbariumHead(nn.Module):
    """Classify a trait using backbone output & phylogenetic orders as sidecar data."""

    def __init__(self, orders: list[str], backbone: str):
        super().__init__()

        self.orders = orders

        model_params = BACKBONES[backbone]

        in_feat = model_params["in_feat"] + len(orders)
        fc_feat1 = in_feat // 4
        fc_feat2 = in_feat // 16

        self.model = nn.Sequential(
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=in_feat, out_features=fc_feat1, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat1),
            #
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=fc_feat1, out_features=fc_feat2, bias=False),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat2),
            #
            # nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=fc_feat2, out_features=1),
            # nn.Sigmoid(),
        )

    def forward(self, x0, x1):
        """Run the classifier forwards with a phylogenetic order (one-hot)."""
        x = torch.cat((x0, x1), dim=1)
        x = self.model(x)
        return x


class HerbariumModel(nn.Module):
    """The herbarium model for a single trait."""

    def __init__(self, orders: list[str], backbone: str, load_model: Path):
        super().__init__()

        model_params = BACKBONES[backbone]
        self.size = model_params["size"]
        self.mean = model_params.get("mean", IMAGENET_MEAN)
        self.std_dev = model_params.get("std_dev", IMAGENET_STD_DEV)

        self.backbone = HerbariumBackbone(backbone)
        self.head = HerbariumHead(orders, backbone)

        self.state = torch.load(load_model) if load_model else {}
        if self.state.get("model_state"):
            logging.info("Loading the model.")
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0, x1):
        """feed the backbone to all of the classifiers."""
        x0 = self.backbone(x0)
        x = self.head(x0, x1)
        return x
