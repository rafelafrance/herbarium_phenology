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
        augment=False,
        normalize=True,
    ) -> None:
        super().__init__()
        self.sheets: list[tuple] = [(s["path"], self.to_classes(s)) for s in sheets]
        self.transform = self.build_transforms(net, augment, normalize)

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
        return image, sheet[1]

    def to_classes(self, sheet):
        """Convert sheet flags to classes."""
        return torch.Tensor([1.0 if sheet[c] == "1" else 0.0 for c in self.all_classes])

    def pos_weight(self):
        """Calculate the positive weight for classes in this dataset."""
        weights = []
        for i in range(len(self.all_classes)):
            weights.append(sum(s[1][i] for s in self.sheets))

        pos_wt = [(len(self) - w) / w for w in weights]
        return torch.Tensor(pos_wt)
