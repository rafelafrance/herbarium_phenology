"""Generate training data."""
import warnings
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from herbarium.pylib.const import ROOT_DIR

Sheet = namedtuple("Sheet", "path coreid order target")


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
            sheets.append(
                Sheet(
                    rec["path"],
                    rec["coreid"],
                    self.to_order(self.orders, rec),
                    # torch.tensor(rec["target"], dtype=torch.float),  # for hydra
                    torch.tensor([rec["target"]], dtype=torch.float),
                )
            )
        return sheets

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
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        xform += [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
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
