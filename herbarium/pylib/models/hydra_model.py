"""This model feed the backbone to all heads, one for each trait."""
from pathlib import Path

import torch
from torch import nn

from . import model_utils
from ..consts import TRAIT_2_INT
from ..consts import TRAITS
from .base_model import BaseBackbone
from .base_model import BaseHead


class HydraModel(nn.Module):
    """The model with every trait getting its own head."""

    def __init__(
        self, orders: list[str], backbone: str, load_model: Path, trait: str = None
    ):
        super().__init__()

        model_utils.get_backbone_params(self, backbone)

        self.backbone = BaseBackbone(backbone)
        self.heads = nn.ModuleList([BaseHead(orders, backbone) for _ in TRAITS])

        self.count = len(TRAITS)
        if trait:
            self.use_head = [False] * self.count
            self.use_head[TRAIT_2_INT[trait]] = True
        else:
            self.use_head = [True] * self.count

        model_utils.load_model_state(self, load_model)

    def forward(self, x0, x1):
        """feed the backbone to all of the classifier heads we're using."""
        x0 = self.backbone(x0)

        xs = torch.zeros((x0.size(0), self.count))
        for i, (head, use) in enumerate(zip(self.heads, self.use_head)):
            if use:
                x = head(x0, x1)
                xs[:, i] = x[:, 0]

        return xs
