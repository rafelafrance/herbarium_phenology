"""Generate training data."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HerbariumDataset(Dataset):
    """Generate augmented data."""

    all_classes = "flowering not_flowering fruiting not_fruiting".split()

    default_size = (380, 380)
    default_mean = [0.7743, 0.7529, 0.7100]
    # default_std_dev = [0.2250, 0.2326, 0.2449]  # 224 x 224
    default_std_dev = [0.2286, 0.2365, 0.2492]  # 380 x 380

    def __init__(
        self, sheets: list[dict],
        augment=False,
        size=None,
        mean=None,
        std_dev=None,
    ) -> None:
        super().__init__()

        size = size if size else self.default_size
        mean = mean if mean else self.default_mean
        std_dev = std_dev if std_dev else self.default_std_dev

        self.sheets: list[tuple] = [(s["path"], self.to_classes(s)) for s in sheets]

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.PILToTensor(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean, std_dev),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.PILToTensor(),
                transforms.Normalize(mean, std_dev),
            ])

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        sheet = self.sheets[index]
        image = Image.open(sheet[0], mode="RGB")
        image = self.transform(image)
        return image, sheet[1]

    def to_classes(self, sheet):
        """Convert sheet flags to classes."""
        return [1.0 if sheet[c] == 1 else 0.0 for c in self.all_classes]
