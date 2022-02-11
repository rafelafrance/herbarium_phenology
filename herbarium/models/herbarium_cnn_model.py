"""Create a model that uses a CNN as a head."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import model_utils
from .backbones import BACKBONES
from .herbarium_model import HerbariumBackbone


class HerbariumCnnHead(nn.Module):
    """Classify a trait using backbone output & phylogenetic orders as input."""

    def __init__(self, orders: list[str], backbone: str):
        super().__init__()

        self.orders = orders

        model_params = BACKBONES[backbone]

        in_feat = model_params["in_feat"]
        cnn_chan1 = 32
        cnn_chan2 = 64
        mix_feat = in_feat + len(orders)
        fc_feat1 = in_feat // 4
        fc_feat2 = in_feat // 16

        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_chan1, 3, padding=1, bias=False),
            nn.BatchNorm1d(cnn_chan1),
            nn.SiLU(inplace=True),
            #
            nn.Conv1d(cnn_chan1, cnn_chan2, 3, padding=1, bias=False),
            nn.BatchNorm1d(cnn_chan2),
            nn.SiLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=mix_feat, out_features=fc_feat1, bias=False),
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
        x0 = torch.unsqueeze(x0, dim=1)
        x0 = self.conv(x0)
        x0 = x0.permute(0, 2, 1)
        x0 = F.adaptive_avg_pool1d(x0, 1).squeeze()
        x = torch.cat((x0, x1), dim=1)
        x = self.fc(x)
        return x


class HerbariumCnnModel(nn.Module):
    """The full model."""

    def __init__(self, orders: list[str], backbone: str, load_model: Path):
        super().__init__()

        model_utils.get_backbone_params(self, backbone)

        self.backbone = HerbariumBackbone(backbone)
        self.head = HerbariumCnnHead(orders, backbone)

        model_utils.load_model_state(self, load_model)

    def forward(self, x0, x1):
        """feed the backbone to all of the classifiers."""
        x0 = self.backbone(x0)
        x = self.head(x0, x1)
        return x
