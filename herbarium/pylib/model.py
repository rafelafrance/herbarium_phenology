import torchvision
from torch import nn

from herbarium.pylib.herbarium_dataset import HerbariumDataset


def freeze(model):
    """Freeze the layers in the model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    """Freeze the layers in the model."""
    for param in model.parameters():
        param.requires_grad = True


class EfficientNetB0:
    """A class for training efficient net models."""

    size = (224, 224)
    mean = [0.7743, 0.7529, 0.7100]
    std_dev = [0.2250, 0.2326, 0.2449]

    @staticmethod
    def get_model():
        """Get the model to use."""
        model = torchvision.models.efficientnet_b0(pretrained=True)
        freeze(model)
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=1280),
            nn.Linear(in_features=1280, out_features=480),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=480),
            nn.Linear(in_features=480, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=len(HerbariumDataset.all_classes)),
        )
        return model


class EfficientNetB4:
    """A class for training efficient net models."""

    size = (380, 380)
    mean = [0.7743, 0.7529, 0.7100]
    std_dev = [0.2286, 0.2365, 0.2492]

    @staticmethod
    def get_model():
        """Get the model to use."""
        model = torchvision.models.efficientnet_b4(pretrained=True)
        freeze(model)
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=1792),
            nn.Linear(in_features=1792, out_features=625),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=625),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=len(HerbariumDataset.all_classes)),
        )
        return model


MODELS = {
    "b0": EfficientNetB0,
    "b4": EfficientNetB4,
}
