"""Generate training data."""
import warnings
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .const import ROOT_DIR

Sheet = namedtuple("Sheet", "path order trait")


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
        trait: str = "flowering",
    ) -> None:
        super().__init__()

        self.trait_name = trait

        self.orders: dict[str, int] = {}
        self.orders_len = 0
        if orders:
            self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}
            self.orders_len = len(orders)

        self.transform = self.build_transforms(net, augment)

        self.sheets: list[Sheet] = []
        for sheet in sheets:
            self.sheets.append(
                Sheet(
                    sheet["path"],
                    self.to_order(sheet),
                    self.to_trait(sheet),
                )
            )

    @staticmethod
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

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(ROOT_DIR / sheet.path).convert("RGB")
            image = self.transform(image)
        return image, sheet.order, sheet.trait

    def to_trait(self, sheet) -> torch.Tensor:
        """Convert sheet flags to trait classes."""
        trait = 1.0 if sheet[self.trait_name] == "1" else 0.0
        return torch.Tensor([trait])

    def to_order(self, sheet):
        """Convert sheet order to a one-hot encoding for the order."""
        order = torch.zeros(self.orders_len, dtype=torch.float)
        order[self.orders[sheet["order_"]]] = 1.0
        return order

    def pos_weight(self) -> torch.Tensor:
        """Calculate the positive weight for trait in this dataset."""
        weight = sum(s.trait for s in self.sheets)
        pos_wt = (len(self) - weight) / weight if weight else 0.0
        return torch.Tensor(pos_wt)
