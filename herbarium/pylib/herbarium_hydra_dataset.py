"""Generate training data."""
import warnings
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .const import ROOT_DIR

Sheet = namedtuple("Sheet", "path coreid order flags traits")


class HerbariumHydraDataset(Dataset):
    """Generate augmented data."""

    all_traits: list[str] = " flowering fruiting leaf_out ".split()

    def __init__(
        self,
        sheets: list[dict],
        net,
        *,
        orders: list[str] = None,
        traits: list[str] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.traits: list[str] = traits if traits else [self.all_traits[0]]

        orders = orders if orders else []
        self.orders: dict[str, int] = {o: i for i, o in enumerate(orders)}

        self.transform = self.build_transforms(net, augment)

        self.sheets: list[Sheet] = []
        for sheet in sheets:
            trait_flags, trait_values = self.to_trait(sheet)
            self.sheets.append(
                Sheet(
                    sheet["path"],
                    sheet["coreid"],
                    self.to_order(sheet),
                    trait_flags,
                    trait_values,
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
        return {
            "image": image,
            "order": sheet.order,
            "flags": sheet.flags,
            "traits": sheet.traits,
            "coreid": sheet.coreid,
        }

    def to_trait(self, sheet) -> (torch.tensor, torch.tensor):
        """Convert sheet flags to trait classes."""
        flags = [0] * len(self.traits)
        traits = [0.0] * len(self.traits)
        for i, trait in enumerate(self.traits):
            flags[i] = 1 if sheet[trait] == "1" or sheet[f"not_{trait}"] == "1" else 0
            traits[i] = 1 if sheet[trait] == "1" else 0
        return torch.tensor(flags), torch.tensor(traits)

    def to_order(self, sheet):
        """Convert the phylogenetic order to a one-hot encoding for the order."""
        order = torch.zeros(len(self.orders), dtype=torch.float)
        order[self.orders[sheet["order_"]]] = 1.0
        return order

    def pos_weight(self) -> torch.tensor:
        """Calculate the positive weight for traits in this dataset."""
        pos = [0.0] * len(self.traits)
        total = [0.0] * len(self.traits)
        for i, _ in enumerate(self.traits):
            total[i] = sum(1.0 if s.flags[i] else 0.0 for s in self.sheets)
            pos[i] = sum(1.0 if s.traits[i] == 1.0 else 0.0 for s in self.sheets)
        pos_wt = [(t - p) / p if p else 0.0 for p, t in zip(pos, total)]
        return torch.tensor(pos_wt)
