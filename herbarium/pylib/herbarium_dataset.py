"""Generate training data."""
import warnings

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HerbariumDataset(Dataset):
    """Generate augmented data."""

    all_classes = "flowering not_flowering fruiting not_fruiting".split()

    default_mean = [0.7743, 0.7529, 0.7100]

    # B0
    default_size = (224, 224)
    default_std_dev = [0.2250, 0.2326, 0.2449]  # 224 x 224

    # B4
    # default_size = (380, 380)
    # default_std_dev = [0.2286, 0.2365, 0.2492]  # 380 x 380

    def __init__(
        self,
        sheets: list[dict],
        augment=False,
        size=None,
        mean=None,
        std_dev=None,
    ) -> None:
        super().__init__()

        size = size if size else self.default_size

        mean = mean if mean else self.default_mean
        mean = torch.Tensor(mean)

        std_dev = std_dev if std_dev else self.default_std_dev
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
