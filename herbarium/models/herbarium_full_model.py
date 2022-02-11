"""A single head per model but we are training the entire network not just the head."""
import copy
import logging
from pathlib import Path

import torch
from torch import nn

from .backbone_params import BACKBONES
from .herbarium_model import HerbariumHead

# Because we are training the entire model we are no longer using ImageNet parameters
FULL_BACKBONES = copy.deepcopy(BACKBONES)

FULL_BACKBONES["b0"]["mean"] = [0.7743, 0.7529, 0.7100]
FULL_BACKBONES["b0"]["std_dev"] = [0.2250, 0.2326, 0.2449]

FULL_BACKBONES["b1"]["mean"] = [0.7743, 0.7529, 0.7100]
FULL_BACKBONES["b1"]["std_dev"] = [0.2250, 0.2326, 0.2449]  # TODO

FULL_BACKBONES["b2"]["mean"] = [0.7743, 0.7529, 0.7100]
FULL_BACKBONES["b2"]["std_dev"] = [0.2250, 0.2326, 0.2449]  # TODO

FULL_BACKBONES["b3"]["mean"] = [0.7743, 0.7529, 0.7100]
FULL_BACKBONES["b3"]["std_dev"] = [0.2286, 0.2365, 0.2492]  # TODO

FULL_BACKBONES["b4"]["mean"] = [0.7743, 0.7529, 0.7100]
FULL_BACKBONES["b4"]["std_dev"] = [0.2286, 0.2365, 0.2492]

FULL_BACKBONES["b7"]["mean"] = [0.7743, 0.7529, 0.7100]
FULL_BACKBONES["b7"]["std_dev"] = [0.2286, 0.2365, 0.2492]  # TODO


class HerbariumFullBackbone(nn.Module):
    """Pretrained backbone."""

    def __init__(self, backbone: str, freeze: bool = False):
        super().__init__()

        model_params = FULL_BACKBONES[backbone]

        self.model = model_params["backbone"](pretrained=False)
        self.model.classifier = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = freeze

    def forward(self, x):
        """Build image output."""
        return self.model(x)


class HerbariumFullModel(nn.Module):
    """The full hydra model."""

    def __init__(self, orders: list[str], backbone: str, load_model: Path):
        super().__init__()

        model_params = FULL_BACKBONES[backbone]
        self.size = model_params["size"]
        self.mean = model_params["mean"]
        self.std_dev = model_params["std_dev"]

        self.backbone = HerbariumFullBackbone(backbone)
        self.head = HerbariumHead(orders, backbone)

        self.state = torch.load(load_model) if load_model else {}
        if self.state.get("model_state"):
            logging.info("Loading a model.")
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0, x1):
        """feed the backbone to all of the classifiers."""
        x0 = self.backbone(x0)
        x = self.head(x0, x1)
        return x
