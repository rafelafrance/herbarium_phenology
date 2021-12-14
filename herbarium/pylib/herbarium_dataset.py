"""Generate training data."""
import warnings

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .const import ROOT_DIR


class HerbariumDataset(Dataset):
    """Generate augmented data."""

    all_classes = "flowering not_flowering fruiting not_fruiting".split()

    def __init__(
        self,
        sheets: list[dict],
        net,
        *,
        orders: list[str] = None,
        augment: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.orders: list[str] = orders
        self.transform = self.build_transforms(net, augment, normalize)

        self.sheets: list[tuple] = []
        for sheet in sheets:
            self.sheets.append(
                (
                    sheet["path"],
                    self.to_order(sheet),
                    self.to_classes(sheet),
                )
            )

    @staticmethod
    def build_transforms(net, augment, normalize):
        """Build a pipeline of image transforms specific to the dataset."""
        xform = [transforms.Resize(net.size)]

        if augment:
            xform += [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        xform += [transforms.ToTensor()]

        if normalize:
            xform += [transforms.Normalize(net.mean, net.std_dev)]

        return transforms.Compose(xform)

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(ROOT_DIR / sheet[0]).convert("RGB")
            image = self.transform(image)
        return image, sheet[1], sheet[2]

    def to_classes(self, sheet):
        """Convert sheet flags to classes."""
        return torch.Tensor([1.0 if sheet[c] == "1" else 0.0 for c in self.all_classes])

    def to_order(self, sheet):
        """Convert sheet order to a one-hot encoding for the order."""
        return torch.Tensor([1.0 if sheet["order_"] == x else 0.0 for x in self.orders])

    def pos_weight(self):
        """Calculate the positive weight for classes in this dataset."""
        weights = []
        for i in range(len(self.all_classes)):
            weights.append(sum(s[1][i] for s in self.sheets))

        pos_wt = [(len(self) - w) / w for w in weights]
        return torch.Tensor(pos_wt)
