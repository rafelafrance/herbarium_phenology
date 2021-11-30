"""Generate training data."""

# import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DataLoader(Dataset):
    """Generate augmented data."""

    all_classes = {
        "none": 0,
        "flowering": 1,
        "not_flowering": 2,
        "fruiting": 3,
        "not_fruiting": 4,
        # "leaf_out": 5,
        # "not_leaf_out": 6,
    }

    def __init__(self, sheets: list[dict], augment=False) -> None:
        super().__init__()
        self.sheets: list[tuple] = [(s["path"], self.to_classes(s)) for s in sheets]
        self.augment = augment

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        sheet = self.sheets[index]
        image = Image.open(sheet[0])
        image = transforms.ToTensor()(image)
        return image, sheet[1]

    def to_classes(self, sheet):
        """Convert sheet flags to classes."""
        classes = [v for k, v in self.all_classes.items() if sheet[k] == 1]
        classes = classes if classes else [0]
        return classes
