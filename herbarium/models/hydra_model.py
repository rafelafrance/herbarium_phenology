"""This model feed the backbone to all heads, one for each trait."""
from pathlib import Path

import torch
from torch import nn

from . import model_util
from .herbarium_model import HerbariumBackbone
from .herbarium_model import HerbariumHead
from herbarium.pylib.const import TRAIT_2_INT
from herbarium.pylib.const import TRAITS


class HydraModel(nn.Module):
    """The model with every trait getting its own head."""

    def __init__(
        self, orders: list[str], backbone: str, load_model: Path, trait: str = None
    ):
        super().__init__()

        model_util.get_backbone_params(self, backbone)

        self.backbone = HerbariumBackbone(backbone)
        self.heads = nn.ModuleList([HerbariumHead(orders, backbone) for _ in TRAITS])

        self.count = len(TRAITS)
        if trait:
            self.use_head = [False] * self.count
            self.use_head[TRAIT_2_INT[trait]] = True
        else:
            self.use_head = [True] * self.count

        model_util.load_model_state(self, load_model)

    def forward(self, x0, x1):
        """feed the backbone to all of the classifier heads we're using."""
        x0 = self.backbone(x0)

        xs = torch.zeros((x0.size(0), self.count))
        for i, (head, use) in enumerate(zip(self.heads, self.use_head)):
            if use:
                x = head(x0, x1)
                xs[:, i] = x[:, 0]

        return xs
