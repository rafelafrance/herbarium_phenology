import logging
from pathlib import Path

import torch
from torch import nn

from herbarium.models.backbone_params import IMAGENET_MEAN
from herbarium.models.backbone_params import IMAGENET_STD_DEV
from herbarium.models.herbarium_full_model import BACKBONES
from herbarium.models.herbarium_model import HerbariumBackbone as HydraBackbone
from herbarium.models.herbarium_model import HerbariumHead as HydraHead
from herbarium.pylib.const import TRAIT_2_INT
from herbarium.pylib.const import TRAITS


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

        self.backbone = HydraBackbone(backbone)
        self.heads = nn.ModuleList([HydraHead(orders, backbone) for _ in TRAITS])

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
