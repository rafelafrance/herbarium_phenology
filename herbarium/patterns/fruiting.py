"""Find fruiting notations."""
from spacy import registry
from traiter.patterns.matcher_patterns import MatcherPatterns

from herbarium.pylib.const import COMMON_PATTERNS

FRUITING = MatcherPatterns(
    "fruiting",
    on_match="herbarium_phenology.fruiting.v1",
    decoder=COMMON_PATTERNS | {},
    patterns=[],
)


@registry.misc(FRUITING.on_match)
def fruiting(ent):
    """Enrich a phrase match."""
