"""Define literals used in the system."""
import os
from pathlib import Path

from traiter.terms.csv_ import Csv

# #########################################################################
# Locations
CURR_DIR = Path(os.getcwd())
IS_SUBDIR = CURR_DIR.name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"
VOCAB_DIR = ROOT_DIR / "herbarium" / "vocabulary"

# #########################################################################
# Term related constants
TERMS = Csv.read_csv(VOCAB_DIR / "herbarium.csv")

# #########################################################################
# Trait related constants
TRAITS = " flowering fruiting leaf_out ".split()
TRAIT_2_INT = {t: i for i, t in enumerate(TRAITS)}
TRAIT_2_STR = {i: t for i, t in enumerate(TRAITS)}
