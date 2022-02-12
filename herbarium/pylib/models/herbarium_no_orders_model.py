"""Create EfficientNets."""
from pathlib import Path

from torch import nn

from . import model_utils
from .backbones import BACKBONES
from .herbarium_model import HerbariumBackbone


class HerbariumNoOrdersHead(nn.Module):
    """Classify a trait using backbone output as input."""

    def __init__(self, orders: list[str], backbone: str):
        super().__init__()

        self.orders = orders

        model_params = BACKBONES[backbone]

        in_feat = model_params["in_feat"]
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

    def forward(self, x):
        """Run the classifier forwards."""
        x = self.model(x)
        return x


class HerbariumNoOrdersModel(nn.Module):
    """The utils model for a single trait."""

    def __init__(
        self, orders: list[str], backbone: str, load_model: Path, trait: str = None
    ):
        super().__init__()

        self.trait = trait

        model_utils.get_backbone_params(self, backbone)

        self.backbone = HerbariumBackbone(backbone)
        self.head = HerbariumNoOrdersHead(orders, backbone)

        model_utils.load_model_state(self, load_model)

    def forward(self, x, _):
        """Feed the backbone to all of the classifiers."""
        x = self.backbone(x)
        x = self.head(x)
        return x
