"""Define literals used in the system."""
import os
from pathlib import Path

from traiter.terms.csv_ import Csv

CURR_DIR = Path(os.getcwd())
IS_SUBDIR = CURR_DIR.name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"
VOCAB_DIR = ROOT_DIR / "phenology" / "vocabulary"

# #########################################################################
# Term related constants
TERMS = Csv.read_csv(VOCAB_DIR / "phenology.csv")
