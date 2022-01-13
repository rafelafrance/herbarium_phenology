"""Override EfficientNet so that it uses multiple inputs on the forward pass."""
from pathlib import Path

import torch
from torch import nn
from torch import Tensor

from .efficient_net_hydra import BACKBONES


class EfficientNetOld(nn.Module):
    """Override EfficientNet so that it uses multiple inputs on the forward pass."""

    def __init__(
        self,
        backbone: str,
        orders: list[str],
        load_weights: Path,
    ):
        super().__init__()

        params = BACKBONES[backbone]
        self.size = params["size"]
        self.mean = (0.485, 0.456, 0.406)  # ImageNet
        self.std_dev = (0.229, 0.224, 0.225)  # ImageNet

        mid_feat = [params["in_feat"] // (2 ** i) for i in range(2, 5)]
        mix_feat = mid_feat[0] + len(orders)

        self.backbone = params["backbone"](pretrained=True)

        # Freeze the top of a pre-trained model
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=params["in_feat"], out_features=mid_feat[0]),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=mid_feat[0]),
        )

        self.multi_classifier = nn.Sequential(
            nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=mix_feat, out_features=mid_feat[1]),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=mid_feat[1]),
            #
            nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=mid_feat[1], out_features=mid_feat[2]),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=mid_feat[2]),
            #
            # nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=mid_feat[2], out_features=1),
            # nn.Sigmoid(),
        )

        self.state = torch.load(load_weights) if load_weights else {}
        if self.state.get("model_state"):
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Run the classifier forwards."""
        x0 = self.backbone(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.multi_classifier(x)
        return x
