"""Generate inference data."""
from collections import namedtuple

from .herbarium_dataset import HerbariumDataset

InferenceSheet = namedtuple("InferenceSheet", "path coreid order")


class InferenceDataset(HerbariumDataset):
    """Create a dataset from images in a directory."""

    def build_sheets(self, image_recs) -> list[InferenceSheet]:
        """Build the sheets used for inference."""
        sheets: list[InferenceSheet] = []
        for rec in image_recs:
            order = self.to_order(self.orders, rec)
            sheets.append(InferenceSheet(rec["path"], rec["coreid"], order))
        return sheets

    def __getitem__(self, index):
        image, sheet = self.raw_item(index)
        return image, sheet.order, sheet.coreid
