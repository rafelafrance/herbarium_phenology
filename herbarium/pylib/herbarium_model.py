"""Create EfficientNets that use plant orders as well as images for input."""
import logging
from pathlib import Path

import torch
import torchvision
from torch import nn

from .const import TRAIT_2_INT
from .const import TRAITS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

BACKBONES = {
    "b0": {
        "backbone": torchvision.models.efficientnet_b0,
        "size": (224, 224),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b1": {
        "backbone": torchvision.models.efficientnet_b1,
        "size": (240, 240),
        "dropout": 0.2,
        "in_feat": 1280,
    },
    "b2": {
        "backbone": torchvision.models.efficientnet_b2,
        "size": (260, 260),
        "dropout": 0.3,
        "in_feat": 1408,
    },
    "b3": {
        "backbone": torchvision.models.efficientnet_b3,
        "size": (300, 300),
        "dropout": 0.3,
        "in_feat": 1536,
    },
    "b4": {
        "backbone": torchvision.models.efficientnet_b4,
        "size": (380, 380),
        "dropout": 0.4,
        "in_feat": 1792,
    },
    # b5: {"size": (456, 456), }
    # b6: {"size": (528, 528), }
    "b7": {
        "backbone": torchvision.models.efficientnet_b7,
        "size": (600, 600),
        "dropout": 0.5,
        "in_feat": 2560,
    },
}


class HerbariumBackbone(nn.Module):
    """Backbone for all of the trait nets."""

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


class HydraModel(nn.Module):
    """The model with every trait getting its own head."""

    def __init__(
        self, orders: list[str], backbone: str, load_model: Path, trait: str = None
    ):
        super().__init__()

        model_params = BACKBONES[backbone]
        self.size = model_params["size"]
        self.mean = model_params.get("mean", IMAGENET_MEAN)
        self.std_dev = model_params.get("std_dev", IMAGENET_STD_DEV)

        self.backbone = HerbariumBackbone(backbone)
        self.heads = nn.ModuleList([HerbariumHead(orders, backbone) for _ in TRAITS])

        self.count = len(TRAITS)
        if trait:
            self.use_head = [False] * self.count
            self.use_head[TRAIT_2_INT[trait]] = True
        else:
            self.use_head = [True] * self.count

        self.state = torch.load(load_model) if load_model else {}
        if self.state.get("model_state"):
            logging.info("Loading the model.")
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0, x1):
        """feed the backbone to all of the classifier heads we're using."""
        x0 = self.backbone(x0)

        xs = torch.zeros((x0.size(0), self.count))
        for i, (head, use) in enumerate(zip(self.heads, self.use_head)):
            if use:
                x = head(x0, x1)
                xs[:, i] = x[:, 0]

        return xs
