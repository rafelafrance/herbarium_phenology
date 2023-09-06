import spacy
from traiter.pylib.pipes import extensions
from traiter.pylib.pipes import tokenizer

from . import terms


# from traiter.pipes.debug import debug_tokens


def pipeline():
    extensions.add_extensions()

    nlp = spacy.load(
        "en_core_web_sm",
        exclude=["ner", "lemmatizer", "tok2vec"],
    )

    tokenizer.setup_tokenizer(nlp)

    terms.build(nlp)

    return nlp
