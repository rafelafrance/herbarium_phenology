from spacy.language import Language
from spacy.util import registry
from traiter.pylib.pattern_compiler import Compiler
from traiter.pylib.pipes import add

from ..consts import ROOT_DIR

VOCAB_DIR = ROOT_DIR / "herbarium" / "vocabulary"

TERMS = VOCAB_DIR / "herbarium.csv"


def build(nlp: Language):
    add.term_pipe(nlp, name="herbarium_terms", path=TERMS)
    add.trait_pipe(nlp, name="herbarium_patterns", compiler=herbarium_patterns())
    # add.debug_tokens(nlp)  # ##########################################


def herbarium_patterns():
    decoder = {
        "flowering": {"ENT_TYPE": "flowering"},
        "flowering_fruiting": {"ENT_TYPE": "flowering_fruiting"},
        "fruiting": {"ENT_TYPE": "fruiting"},
        "leaf_out": {"ENT_TYPE": "leaf_out"},
        "not_flowering": {"ENT_TYPE": "not_flowering"},
        "not_fruiting": {"ENT_TYPE": "not_fruiting"},
        "not_leaf_out": {"ENT_TYPE": "not_leaf_out"},
    }
    return [
        Compiler(
            label="term",
            on_match="term_match",
            keep="term",
            decoder=decoder,
            patterns=[
                " flowering+ ",
                " flowering_fruiting+ ",
                " fruiting+ ",
                " leaf_out+ ",
                " not_flowering+ ",
                " not_fruiting+ ",
                " not_leaf_out+ ",
            ],
        ),
    ]


@registry.misc("term_match")
def term_match(ent):
    relabel = next(t._.term for t in ent)
    ent._.relabel = relabel
    ent._.data = {relabel: ent.text}
