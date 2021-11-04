"""Create a trait pipeline."""
import spacy
from traiter.patterns.matcher_patterns import add_ruler_patterns
from traiter.tokenizer_util import append_tokenizer_regexes

from herbarium.patterns.flowering import FLOWERING
from herbarium.patterns.fruiting import FRUITING
from herbarium.patterns.leaf_out import LEAF_OUT
from herbarium.pylib.const import TERMS
# from traiter.patterns.matcher_patterns import as_dicts
# from traiter.patterns.matcher_patterns import patterns_to_dispatch
# from traiter.pipes.add_entity_data import ADD_ENTITY_DATA
# from traiter.pipes.cleanup import CLEANUP
# # from traiter.pipes.debug import DEBUG_ENTITIES, DEBUG_TOKENS
# from traiter.pipes.dependency import DEPENDENCY
# from traiter.pipes.sentence import SENTENCE
# from traiter.pipes.simple_entity_data import SIMPLE_ENTITY_DATA
# from traiter.pipes.update_entity_data import UPDATE_ENTITY_DATA
# from traiter.tokenizer_util import append_abbrevs

TERM_RULES = [FLOWERING, FRUITING, LEAF_OUT]

# ADD_DATA = [
#     COLOR, MARGIN_SHAPE, N_SHAPE, SHAPE, PART_AS_LOCATION, SUBPART_AS_LOCATION]


def pipeline():
    """Create a pipeline for extracting traits."""
    nlp = spacy.load("en_core_web_sm", exclude=["", "ner"])
    append_tokenizer_regexes(nlp)

    # Add a pipe to identify phrases and patterns as base-level traits.
    config = {"phrase_matcher_attr": "LOWER"}
    term_ruler = nlp.add_pipe(
        "entity_ruler", name="term_ruler", config=config, before="parser"
    )
    term_ruler.add_patterns(TERMS.for_entity_ruler())
    add_ruler_patterns(term_ruler, TERM_RULES)

    # nlp.add_pipe('merge_entities', name='term_merger')
    # nlp.add_pipe(SIMPLE_ENTITY_DATA, after='term_merger', config={'replace': REPLACE})
    #
    # config = {'patterns': as_dicts(UPDATE_DATA)}
    # nlp.add_pipe(UPDATE_ENTITY_DATA, name='update_entities', config=config)

    # Add a pipe to group tokens into larger traits
    # config = {'overwrite_ents': True}
    # match_ruler = nlp.add_pipe('entity_ruler', name='match_ruler', config=config)
    # add_ruler_patterns(match_ruler, ADD_DATA)
    #
    # nlp.add_pipe(ADD_ENTITY_DATA, config={'dispatch': patterns_to_dispatch(ADD_DATA)})
    #
    # nlp.add_pipe(CLEANUP, config={'entities': FORGET})

    # nlp.add_pipe(DEBUG_TOKENS, config={'message': ''})

    # config = {'patterns': as_dicts(LINKERS)}
    # nlp.add_pipe(DEPENDENCY, name='part_linker', config=config)

    return nlp
