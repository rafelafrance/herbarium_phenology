"""Model architectures used."""
from typing import Optional

import torch
import torchvision
from torch import nn
from torchvision.models import EfficientNet

from .herbarium_dataset import HerbariumDataset


class Classifier:
    """The mixed-input classifier layer for an EfficientNet."""

    def __init__(self, net, orders):
        mix_feat = net.feat[1] + len(orders)
        out_feat = len(HerbariumDataset.all_classes)

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=net.in_feat, out_features=net.feat[0]),
            nn.BatchNorm1d(num_features=net.feat[0]),
            nn.SiLU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=mix_feat, out_features=net.feat[1]),
            nn.BatchNorm1d(num_features=net.feat[1]),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=net.feat[1], out_features=net.feat[2]),
            nn.BatchNorm1d(num_features=net.feat[2]),
            nn.SiLU(inplace=True),
            nn.Dropout(p=net.dropout, inplace=True),
            nn.Linear(in_features=net.feat[2], out_features=out_feat),
            # nn.Softmax(dim=1),
        )

    def forward(self, x0, x1):
        """Run the classifier."""
        x0 = self.classifier1(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.classifier2(x)
        return x


class Net:
    """Base class for all the EfficientNet classes."""

    dropout = 0.2
    leaky_relu = 0.2
    in_feat = 1280

    def __init__(self, args):
        self.model: Optional[EfficientNet] = None
        self.database = args.database
        self.freeze = args.freeze
        self.state = torch.load(args.load_weights) if args.load_weights else {}
        self.feat = [self.in_feat // (2 ** i) for i in range(1, 4)]

    def freeze_all(self) -> None:
        """Freeze the layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Freeze the layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = True


class NetB0(Net):
    """A class for training efficient net models."""

    in_feat = 1280
    size = (224, 224)
    mean = [0.7743, 0.7529, 0.7100]
    std_dev = [0.2250, 0.2326, 0.2449]

    def __init__(self, args, orders):
        super().__init__(args)

        self.model = torchvision.models.efficientnet_b0(pretrained=True)

        if self.freeze:
            self.freeze_all()

        self.model.classifier = Classifier(self, orders)

        if self.state.get("model_state"):
            self.model.load_state_dict(self.state["model_state"])


class NetB3(Net):
    """A class for training efficient net models."""

    in_feat = 1536
    size = (300, 300)
    mean = [0.7743, 0.7529, 0.7100]
    std_dev = [0.2286, 0.2365, 0.2492]  # TODO

    def __init__(self, args, orders):
        super().__init__(args)

        self.model = torchvision.models.efficientnet_b3(pretrained=True)

        if self.freeze:
            self.freeze_all()

        self.model.classifier = Classifier(self, orders)

        if self.state.get("model_state"):
            self.model.load_state_dict(self.state["model_state"])


class NetB4(Net):
    """A class for training efficient net models."""

    in_feat = 1792
    size = (380, 380)
    mean = [0.7743, 0.7529, 0.7100]
    std_dev = [0.2286, 0.2365, 0.2492]

    def __init__(self, args, orders):
        super().__init__(args)

        self.model = torchvision.models.efficientnet_b4(pretrained=True)

        if self.freeze:
            self.freeze_all()

        self.model.classifier = Classifier(self, orders)

        if self.state.get("model_state"):
            self.model.load_state_dict(self.state["model_state"])


NETS = {
    "b0": NetB0,
    "b3": NetB3,
    "b4": NetB4,
}
