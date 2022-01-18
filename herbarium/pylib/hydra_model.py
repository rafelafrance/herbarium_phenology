"""Create EfficientNets that uses plant orders as well as images for input."""
from argparse import Namespace

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


class HydraBackbone(nn.Module):
    """Backbone for all of the trait nets."""

    def __init__(self, args: Namespace):
        super().__init__()

        model_params = BACKBONES[args.backbone]
        self.size = model_params["size"]
        self.mean = (0.485, 0.456, 0.406)  # ImageNet
        self.std_dev = (0.229, 0.224, 0.225)  # ImageNet

        self.model = model_params["backbone"](pretrained=True)
        self.model.classifier = nn.Sequential(nn.Identity())

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Build image output."""
        return self.model(x)


class HydraHead(nn.Module):
    """Classify a trait using backbone output & phylogenetic orders as sidecar data."""

    def __init__(self, orders: list[str], args: Namespace):
        super().__init__()

        self.orders = orders

        self.workers = args.workers
        self.batch_size = args.batch_size
        self.lr = args.learning_rate

        model_params = BACKBONES[args.backbone]

        in_feat = model_params["in_feat"] + len(orders)
        fc_feat1 = in_feat // 4
        fc_feat2 = in_feat // 8
        fc_feat3 = in_feat // 16

        self.model = nn.Sequential(
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=in_feat, out_features=fc_feat1),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat1),
            #
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=fc_feat1, out_features=fc_feat2),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat2),
            #
            nn.Dropout(p=model_params["dropout"], inplace=True),
            nn.Linear(in_features=fc_feat2, out_features=fc_feat3),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_features=fc_feat3),
            #
            # nn.Dropout(p=params["dropout"], inplace=True),
            nn.Linear(in_features=fc_feat3, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Run the classifier forwards with a phylogenetic order (one-hot)."""
        return self.model(x)


class HydraModel(nn.Module):
    """The full hydra model."""

    def __init__(self, traits: list[str], orders: list[str], args: Namespace):
        super().__init__()

        self.backbone = HydraBackbone(args)
        self.heads = nn.ModuleList([HydraHead(orders, args) for _ in traits])

        self.state = torch.load(args.load_weights) if args.load_weights else {}
        if self.state.get("model_state"):
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0, x1):
        """feed the backbone to all of the classifiers."""
        x0 = self.backbone(x0)
        x = torch.cat((x0, x1), dim=1)
        xs = [h(x) for h in self.heads]
        return torch.hstack(xs)
