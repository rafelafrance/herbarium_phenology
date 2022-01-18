"""Generate training data."""
import warnings
from collections import namedtuple
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import db
from .const import ROOT_DIR

Sheet = namedtuple("Sheet", "path coreid order trait")
InferenceSheet = namedtuple("Sheet", "path coreid order")


def build_transforms(net, augment):
    """Build a pipeline of image transforms specific to the dataset."""
    xform = [transforms.Resize(net.size)]

    if augment:
        xform += [
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]

    xform += [
        transforms.ToTensor(),
        transforms.Normalize(net.mean, net.std_dev),
    ]

    return transforms.Compose(xform)


def to_order(orders, sheet):
    """Convert the phylogenetic order to a one-hot encoding for the order."""
    order = torch.zeros(len(orders), dtype=torch.float)
    order[orders[sheet["order_"]]] = 1.0
    return order


class HydraDataset(Dataset):
    """Generate augmented data."""

    all_traits: list[str] = " flowering fruiting leaf_out ".split()

    def __init__(
        self,
        sheets: list[dict],
        net,
        *,
        trait_name: str = None,
        orders: list[str] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.traits: str = trait_name if trait_name else self.all_traits[0]

        orders = orders if orders else []
        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = build_transforms(net, augment)

        self.sheets: list[Sheet] = []
        for sheet in sheets:
            self.sheets.append(
                Sheet(
                    sheet["path"],
                    sheet["coreid"],
                    to_order(orders, sheet),
                    self.to_trait(sheet),
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
        return image, sheet.order, sheet.trait, sheet.coreid

    def to_trait(self, sheet) -> (torch.tensor, torch.tensor):
        """Convert sheet traits to trait classes."""
        traits = [0.0] * len(self.traits)
        for i, trait in enumerate(self.traits):
            traits[i] = 1 if sheet[trait] == "1" else 0
        return torch.tensor(traits, dtype=torch.float)

    def pos_weight(self) -> torch.tensor:
        """Calculate the positive weight for traits in this dataset."""
        pos = [0.0] * len(self.traits)
        total = [0.0] * len(self.traits)
        for i, _ in enumerate(self.traits):
            pos[i] = sum(1.0 if s.trait[i] == 1.0 else 0.0 for s in self.sheets)
        pos_wt = [(t - p) / p if p else 0.0 for p, t in zip(pos, total)]
        return torch.tensor(pos_wt)


class InferenceDataset(Dataset):
    """Create a dataset from images in a directory."""

    all_traits = " flowering fruiting leaf_out ".split()

    def __init__(
        self,
        database: Path,
        paths: list[Path],
        net,
        *,
        orders: list[str] = None,
        traits: list[str] = None,
    ) -> None:
        super().__init__()

        self.trait: str = traits if traits else self.all_traits[0]

        orders = orders if orders else []
        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = build_transforms(net)

        rows = {r["coreid"]: r for r in db.select_images(database)}

        self.sheets: list[InferenceSheet] = []
        for path in paths:
            coreid = path.stem
            sheet = rows[coreid]
            self.sheets.append(
                InferenceSheet(
                    str(path),
                    coreid,
                    to_order(orders, sheet),
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
