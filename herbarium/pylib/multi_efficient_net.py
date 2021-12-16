"""Model architectures used."""

import torch
import torchvision
from torch import nn
from torch import Tensor

from .herbarium_dataset import HerbariumDataset


class MultiEfficientNet(nn.Module):
    """Override EfficientNet so that it uses multiple inputs on the forward pass."""

    def __init__(self, efficient_net, in_feat, orders_len, load_weights, freeze):
        super().__init__()
        mid_feat = [in_feat // (2 ** i) for i in range(1, 4)]
        dropout = 0.2
        mix_feat = mid_feat[0] + orders_len
        out_feat = len(HerbariumDataset.all_classes)

        self.efficient_net = efficient_net
        self.efficient_net.classifier = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=mid_feat[0]),
            nn.BatchNorm1d(num_features=mid_feat[0]),
            nn.SiLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=mix_feat, out_features=mid_feat[1]),
            nn.BatchNorm1d(num_features=mid_feat[1]),
            nn.SiLU(inplace=True),
            #
            nn.Linear(in_features=mid_feat[1], out_features=mid_feat[2]),
            nn.BatchNorm1d(num_features=mid_feat[2]),
            nn.SiLU(inplace=True),
            #
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=mid_feat[2], out_features=out_feat),
            # nn.Softmax(dim=1),
        )

        self.state = torch.load(load_weights) if load_weights else {}
        if self.state.get("model_state"):
            self.model.load_state_dict(self.state["model_state"])

        if freeze:
            for param in self.efficient_net.parameters():
                param.requires_grad = False

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Run the classifier forwards."""
        x0 = self.efficient_net(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.classifier(x)
        return x


class MultiEfficientNetB0(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (224, 224)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2250, 0.2326, 0.2449]

        in_feat = 1280
        efficient_net = torchvision.models.efficientnet_b0(pretrained=True)

        super().__init__(efficient_net, in_feat, orders_len, load_weights, freeze)


class MultiEfficientNetB3(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (300, 300)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2286, 0.2365, 0.2492]  # TODO

        in_feat = 1536
        efficient_net = torchvision.models.efficientnet_b3(pretrained=True)

        super().__init__(efficient_net, in_feat, orders_len, load_weights, freeze)


class MultiEfficientNetB4(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (380, 380)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2286, 0.2365, 0.2492]

        in_feat = 1792
        efficient_net = torchvision.models.efficientnet_b3(pretrained=True)

        super().__init__(efficient_net, in_feat, orders_len, load_weights, freeze)


NETS = {
    "b0": MultiEfficientNetB0,
    "b3": MultiEfficientNetB3,
    "b4": MultiEfficientNetB4,
}
