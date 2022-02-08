"""Literals used in the system."""
import os
from pathlib import Path

from traiter.terms.csv_ import Csv

# #########################################################################
CURR_DIR = Path(os.getcwd())
IS_SUBDIR = CURR_DIR.name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"
VOCAB_DIR = ROOT_DIR / "herbarium" / "vocabulary"

# #########################################################################
TERMS = Csv.read_csv(VOCAB_DIR / "herbarium.csv")

# #########################################################################
TRAITS = " flowering fruiting leaf_out ".split()
TRAIT_2_INT = {t: i for i, t in enumerate(TRAITS)}
TRAIT_2_STR = {i: t for i, t in enumerate(TRAITS)}

# #########################################################################
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)
