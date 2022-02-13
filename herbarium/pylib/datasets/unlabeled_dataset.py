"""Generate inference data."""
from collections import namedtuple

from .labeled_dataset import LabeledDataset

UnlabeledSheet = namedtuple("UnlabeledSheet", "path coreid order")


class UnlabeledDataset(LabeledDataset):
    """Create a dataset from images in a directory."""

    def build_sheets(self, image_recs) -> list[UnlabeledSheet]:
        """Build the sheets used for inference."""
        sheets: list[UnlabeledSheet] = []
        for rec in image_recs:
            order = self.to_order(self.orders, rec)
            sheets.append(UnlabeledSheet(rec["path"], rec["coreid"], order))
        return sheets

    def __getitem__(self, index):
        image, sheet = self.raw_item(index)
        return image, sheet.order, sheet.coreid
