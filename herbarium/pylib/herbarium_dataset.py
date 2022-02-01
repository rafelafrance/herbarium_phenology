"""Generate training data."""
import warnings
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .const import ROOT_DIR

Sheet = namedtuple("Sheet", "path coreid order target")
InferenceSheet = namedtuple("InferenceSheet", "path coreid order")


class HerbariumDataset(Dataset):
    """Generate augmented data."""

    def __init__(
        self,
        image_recs: list[dict],
        model,
        *,
        orders: list[str],
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}
        self.transform = self.build_transforms(model, augment)
        self.sheets = self.build_sheets(image_recs)

    def build_sheets(self, image_recs) -> list[Sheet]:
        """Build the sheets we are using for the dataset."""
        sheets: list[Sheet] = []
        for rec in image_recs:
            target = self.get_target(rec)
            sheets.append(
                Sheet(
                    rec["path"],
                    rec["coreid"],
                    self.to_order(self.orders, rec),
                    torch.tensor([target], dtype=torch.float),
                )
            )
        return sheets

    def get_target(self, rec):
        """Return the target value for the sheet."""
        return rec["target"]

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index) -> tuple:
        image, sheet = self.raw_item(index)
        return image, sheet.order, sheet.target, sheet.coreid

    def raw_item(self, index):
        """Get the raw item data."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(ROOT_DIR / sheet.path).convert("RGB")
            image = self.transform(image)
        return image, sheet

    @staticmethod
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

    @staticmethod
    def to_order(orders, rec):
        """Convert the phylogenetic order to a one-hot encoding for the order."""
        order = torch.zeros(len(orders), dtype=torch.float)
        order[orders[rec["order_"]]] = 1.0
        return order

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the trait."""
        pos = sum(s.target for s in self.sheets)
        pos_wt = (len(self) - pos) / pos if pos > 0.0 else 1.0
        return [pos_wt]


class PseudoDataset(HerbariumDataset):
    """Create a dataset for pseudo-label records."""

    def __init__(
        self,
        image_recs: list[dict],
        model,
        *,
        orders: list[str],
        min_threshold: float,
        max_threshold: float,
    ) -> None:
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        super().__init__(image_recs, model, orders=orders, augment=True)

    def __getitem__(self, index):
        image, sheet = self.raw_item(index)
        return image, sheet.order, sheet.target, sheet.coreid

    def get_target(self, rec):
        """Return the target value for the sheet."""
        return 0.0 if rec["pred"] <= self.min_threshold else 1.0


class InferenceDataset(HerbariumDataset):
    """Create a dataset from images in a directory."""

    def build_sheets(self, image_recs) -> list[InferenceSheet]:
        """Build the sheets used for inference."""
        sheets: list[InferenceSheet] = []
        for rec in image_recs:
            sheets.append(
                InferenceSheet(
                    rec["path"], rec["coreid"], self.to_order(self.orders, rec)
                )
            )
        return sheets

    def __getitem__(self, index):
        image, sheet = self.raw_item(index)
        return image, sheet.order, sheet.coreid
