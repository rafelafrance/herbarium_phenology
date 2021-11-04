"""Find flowering notations."""
from spacy import registry
from traiter.patterns.matcher_patterns import MatcherPatterns

from herbarium.pylib.const import COMMON_PATTERNS

FLOWERING = MatcherPatterns(
    "flowering",
    on_match="herbarium_phenology.flowering.v1",
    decoder=COMMON_PATTERNS | {},
    patterns=[],
)


@registry.misc(FLOWERING.on_match)
def flowering(ent):
    """Enrich a phrase match."""
