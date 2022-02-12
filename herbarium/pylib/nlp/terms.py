"""Vocabulary constants."""
from traiter.terms.csv_ import Csv

from ..consts import ROOT_DIR

VOCAB_DIR = ROOT_DIR / "herbarium" / "vocabulary"

TERMS = Csv.read_csv(VOCAB_DIR / "herbarium.csv")
