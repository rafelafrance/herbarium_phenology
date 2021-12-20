"""Model architectures used."""
import torch
import torchvision
from torch import nn
from torch import Tensor

# from .herbarium_dataset import HerbariumDataset
# b0 224, b1 240, b2 260, b3 300, b4 380, b5 456, b6 528, b7 600


class MultiEfficientNet(nn.Module):
    """Override EfficientNet so that it uses multiple inputs on the forward pass."""

    def __init__(self, efficient_net, orders_len, load_weights, freeze):
        super().__init__()

        mid_feat = [self.in_feat // (2 ** i) for i in range(1, 4)]
        mix_feat = mid_feat[0] + orders_len
        out_feat = 1  # len(HerbariumDataset.all_classes // 2)

        self.efficient_net = efficient_net

        # I don't want to freeze the new (randomized) portions of the net, below.
        if freeze:
            for param in self.efficient_net.parameters():
                param.requires_grad = False

        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(in_features=self.in_feat, out_features=mid_feat[0]),
            nn.BatchNorm1d(num_features=mid_feat[0]),
            nn.SiLU(inplace=True),
        )

        self.multi_classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(in_features=mix_feat, out_features=mid_feat[1]),
            nn.BatchNorm1d(num_features=mid_feat[1]),
            nn.SiLU(inplace=True),
            #
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(in_features=mid_feat[1], out_features=mid_feat[2]),
            nn.BatchNorm1d(num_features=mid_feat[2]),
            nn.SiLU(inplace=True),
            #
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(in_features=mid_feat[2], out_features=out_feat),
        )

        self.state = torch.load(load_weights) if load_weights else {}
        if self.state.get("model_state"):
            self.load_state_dict(self.state["model_state"])

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Run the classifier forwards."""
        x0 = self.efficient_net(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.multi_classifier(x)
        return x


class MultiEfficientNetB0(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (224, 224)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2250, 0.2326, 0.2449]

        self.dropout = 0.2
        self.in_feat = 1280
        efficient_net = torchvision.models.efficientnet_b0(pretrained=True)

        super().__init__(efficient_net, orders_len, load_weights, freeze)


class MultiEfficientNetB3(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (300, 300)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2286, 0.2365, 0.2492]  # TODO

        self.dropout = 0.3
        self.in_feat = 1536
        efficient_net = torchvision.models.efficientnet_b3(pretrained=True)

        super().__init__(efficient_net, orders_len, load_weights, freeze)


class MultiEfficientNetB4(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (380, 380)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2286, 0.2365, 0.2492]

        self.dropout = 0.4
        self.in_feat = 1792
        efficient_net = torchvision.models.efficientnet_b4(pretrained=True)

        super().__init__(efficient_net, orders_len, load_weights, freeze)


class MultiEfficientNetB7(MultiEfficientNet):
    """A class for training efficient net models."""

    def __init__(self, orders_len, load_weights, freeze):
        self.size = (600, 600)
        self.mean = [0.7743, 0.7529, 0.7100]
        self.std_dev = [0.2286, 0.2365, 0.2492]  # TODO

        self.dropout = 0.5
        self.in_feat = 2560
        efficient_net = torchvision.models.efficientnet_b7(pretrained=True)

        super().__init__(efficient_net, orders_len, load_weights, freeze)


NETS = {
    "b0": MultiEfficientNetB0,
    "b3": MultiEfficientNetB3,
    "b4": MultiEfficientNetB4,
    "b7": MultiEfficientNetB7,
}
