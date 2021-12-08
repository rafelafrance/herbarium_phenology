"""Generate training data."""
import warnings

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HerbariumDataset(Dataset):
    """Generate augmented data."""

    all_classes = "flowering not_flowering fruiting not_fruiting".split()

    def __init__(
        self,
        sheets: list[dict],
        classifier,
        mean=None,
        std_dev=None,
        augment=False,
    ) -> None:
        super().__init__()

        size = classifier.size

        mean = mean if mean else classifier.default_mean
        mean = torch.Tensor(mean)

        std_dev = std_dev if std_dev else classifier.default_std_dev
        std_dev = torch.Tensor(std_dev)

        self.sheets: list[tuple] = [(s["path"], self.to_classes(s)) for s in sheets]

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev),
            ])

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(sheet[0]).convert("RGB")
            image = self.transform(image)
        return image, sheet[1]

    def to_classes(self, sheet):
        """Convert sheet flags to classes."""
        return torch.Tensor([1.0 if sheet[c] == '1' else 0.0 for c in self.all_classes])

    def pos_weight(self):
        """Calculate the positive weight for classes in this dataset."""
        weights = []
        for i in range(len(self.all_classes)):
            weights.append(sum(s[1][i] for s in self.sheets))

        pos_wt = [(len(self) - w) / w for w in weights]
        return torch.Tensor(pos_wt)
