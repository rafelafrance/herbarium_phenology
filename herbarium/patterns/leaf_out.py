"""Find leaf-out notations."""
from spacy import registry
from traiter.patterns.matcher_patterns import MatcherPatterns

from herbarium.pylib.const import COMMON_PATTERNS

LEAF_OUT = MatcherPatterns(
    "leaf_out",
    on_match="herbarium_phenology.leaf_out.v1",
    decoder=COMMON_PATTERNS | {},
    patterns=[],
)


@registry.misc(LEAF_OUT.on_match)
def leaf_out(ent):
    """Enrich a phrase match."""
