"""Generate training data."""
import warnings
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .const import ROOT_DIR

ALL_TRAITS = " flowering fruiting leaf_out ".split()

Sheet = namedtuple("Sheet", "path coreid order y_true")
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
        trait_name: str = None,
        orders: list[str] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.trait: str = trait_name if trait_name else ALL_TRAITS[0]

        orders = orders if orders else []
        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = build_transforms(model, augment)

        self.sheets: list[Sheet] = []
        for sheet in sheets:
            self.sheets.append(
                Sheet(
                    sheet["path"],
                    sheet["coreid"],
                    to_order(self.orders, sheet),
                    self.y_true(sheet),
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
        return image, sheet.order, sheet.y_true, sheet.coreid

    def y_true(self, sheet) -> (torch.tensor, torch.tensor):
        """Convert sheet traits to trait classes."""
        value = 1.0 if sheet[self.trait] == "1" else 0.0
        return torch.tensor([value], dtype=torch.float)

    def pos_weight(self) -> torch.tensor:
        """Calculate the weights for the positive & negative cases of the trait."""
        total = len(self)
        pos = sum(1.0 if s.y_true == 1.0 else 0.0 for s in self.sheets)
        neg = total - pos
        pos_wt = pos / neg if neg > 0.0 else 1.0
        return torch.tensor(pos_wt, dtype=torch.float)


class InferenceDataset(Dataset):
    """Create a dataset from images in a directory."""

    def __init__(
        self,
        image_recs: list[dict],
        model,
        *,
        trait_name: str = None,
        orders: list[str] = None,
    ) -> None:
        super().__init__()

        self.trait: str = trait_name if trait_name else ALL_TRAITS[0]

        orders = orders if orders else []
        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = build_transforms(model)

        self.records: list[InferenceSheet] = []
        for rec in image_recs:
            self.sheets.append(
                InferenceSheet(
                    rec["path"],
                    rec["coreid"],
                    to_order(self.orders, rec),
                )
            )

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            rec = self.records[index]
            image = Image.open(ROOT_DIR / rec.path).convert("RGB")
            image = self.transform(image)
        return image, rec.order, rec.coreid
