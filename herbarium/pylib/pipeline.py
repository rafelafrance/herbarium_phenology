"""Create a trait pipeline."""
import spacy
from traiter.pipes.add_entity_data import ADD_ENTITY_DATA
# from traiter.pipes.debug import debug_tokens
from traiter.tokenizer_util import append_tokenizer_regexes

from .const import TERMS


def pipeline():
    """Create a pipeline for extracting traits."""
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    append_tokenizer_regexes(nlp)

    # Add a pipe to identify phrases and patterns as base-level traits.
    config = {"phrase_matcher_attr": "LOWER"}
    term_ruler = nlp.add_pipe(
        "entity_ruler", name="term_ruler", config=config, before="parser"
    )
    term_ruler.add_patterns(TERMS.for_entity_ruler())

    nlp.add_pipe(ADD_ENTITY_DATA, config={"dispatch": {}})
    return nlp
