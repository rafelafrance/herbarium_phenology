"""EfficientNet using plant orders and images for input, separate heads per trait."""
from typing import Any

import torch
import torchvision
from torch import nn

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


class EfficientNetHydra(nn.Module):
    """An EfficientNet that uses phylogenetic orders as sidecar data."""

    def __init__(self, traits: list[str], orders: list[str], args: dict[Any]):
        super().__init__()

        self.traits = traits
        self.orders = orders

        params = BACKBONES[args["backbone"]]
        self.size = params["size"]
        self.mean = (0.485, 0.456, 0.406)  # ImageNet
        self.std_dev = (0.229, 0.224, 0.225)  # ImageNet

        in_feat = params["in_feat"] + len(orders)
        fc_feat1 = in_feat // 4
        fc_feat2 = in_feat // 8
        fc_feat3 = in_feat // 16

        self.backbone = params["backbone"](pretrained=True)
        self.backbone.classifier = nn.Sequential(nn.Identity())

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.heads = []
        for _ in traits:
            self.heads.append(
                nn.Sequential(
                    nn.Dropout(p=params["dropout"], inplace=True),
                    nn.Linear(in_features=in_feat, out_features=fc_feat1),
                    nn.SiLU(inplace=True),
                    nn.BatchNorm1d(num_features=fc_feat1),
                    #
                    nn.Dropout(p=params["dropout"], inplace=True),
                    nn.Linear(in_features=fc_feat1, out_features=fc_feat2),
                    nn.SiLU(inplace=True),
                    nn.BatchNorm1d(num_features=fc_feat2),
                    #
                    nn.Dropout(p=params["dropout"], inplace=True),
                    nn.Linear(in_features=fc_feat2, out_features=fc_feat3),
                    nn.SiLU(inplace=True),
                    nn.BatchNorm1d(num_features=fc_feat3),
                    #
                    # nn.Dropout(p=self.dropout, inplace=True),
                    nn.Linear(in_features=fc_feat3, out_features=1),
                    # nn.Sigmoid(),
                )
            )

        self.state = torch.load(args["load_weights"]) if args["load_weights"] else {}
        if self.state.get("model_state"):
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0, x1):
        """Run the classifier forwards through multiple heads."""
        x0 = self.backbone(x0)
        x = torch.cat((x0, x1), dim=1)
        return [h(x) for h in self.heads]
