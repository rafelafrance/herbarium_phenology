"""Generate training data."""
import warnings
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .const import ROOT_DIR

Sheet = namedtuple("Sheet", "path coreid order target")
InferenceSheet = namedtuple("Sheet", "path coreid order")


def build_transforms(model, augment=False):
    """Build a pipeline of image transforms specific to the dataset."""
    xform = [transforms.Resize(model.size)]

    if augment:
        xform += [
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]

    xform += [
        transforms.ToTensor(),
        transforms.Normalize(model.mean, model.std_dev),
    ]

    return transforms.Compose(xform)


def to_order(orders, rec):
    """Convert the phylogenetic order to a one-hot encoding for the order."""
    order = torch.zeros(len(orders), dtype=torch.float)
    order[orders[rec["order_"]]] = 1.0
    return order


class HerbariumDataset(Dataset):
    """Generate augmented data."""

    def __init__(
        self,
        sheets: list[dict],
        model,
        *,
        orders: list[str],
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = build_transforms(model, augment)

        self.sheets: list[Sheet] = []
        for sheet in sheets:
            self.sheets.append(
                Sheet(
                    sheet["path"],
                    sheet["coreid"],
                    to_order(self.orders, sheet),
                    torch.tensor([sheet["target"]], dtype=torch.float),
                )
            )

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index) -> tuple:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(ROOT_DIR / sheet.path).convert("RGB")
            image = self.transform(image)
        return image, sheet.order, sheet.target, sheet.coreid

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the trait."""
        pos = sum(s.target for s in self.sheets)
        pos_wt = (len(self) - pos) / pos if pos > 0.0 else 1.0
        return [pos_wt]


class InferenceDataset(Dataset):
    """Create a dataset from images in a directory."""

    def __init__(
        self,
        image_recs: list[dict],
        model,
        *,
        orders: list[str],
    ) -> None:
        super().__init__()

        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = build_transforms(model)

        self.sheets: list[InferenceSheet] = []
        for sheet in image_recs:
            self.sheets.append(
                InferenceSheet(
                    sheet["path"],
                    sheet["coreid"],
                    to_order(self.orders, sheet),
                )
            )

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(ROOT_DIR / sheet.path).convert("RGB")
            image = self.transform(image)
        return image, sheet.order, sheet.coreid
