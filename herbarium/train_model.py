#!/usr/bin/env python3
"""Train a model to classify herbarium traits."""
import argparse
import textwrap
from pathlib import Path

import torch

from pylib.classifier import EfficientNetB0, EfficientNetB4

CLASSIFIERS = {
    "b0": EfficientNetB0,
    "b4": EfficientNetB4,
}


def parse_args():
    """Process command-line arguments."""
    description = """Train a herbarium phenology classifier."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        "--database", "--db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the SQLite3 database (angiosperm data).""",
    )

    arg_parser.add_argument(
        '--save-model', required=True, help="""Save best models to this path.""")

    arg_parser.add_argument(
        "--classifier",
        choices=list(CLASSIFIERS.keys()),
        default=list(CLASSIFIERS.keys())[0],
        help="""Which EfficientNet classifier model to use.""",
    )

    default = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    arg_parser.add_argument(
        '--device',
        default=default,
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            We'll try to default to either 'cpu' or 'cuda:0' depending on the
            availability of a GPU. (default: %(default)s)""")

    arg_parser.add_argument(
        '--learning-rate', '--lr', type=float, default=0.0005,
        help="""Initial learning rate. (default: %(default)s)""")

    arg_parser.add_argument(
        '--batch-size', type=int, default=16,
        help="""Input batch size. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    arg_parser.add_argument(
        '--prev-model', help="""Use this model.""")

    arg_parser.add_argument(
        '--epochs', type=int, default=100,
        help="""How many epochs to train. (default: %(default)s)""")

    arg_parser.add_argument(
        "--split-run",
        default="first_split",
        help="""Which data split to use. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    classifier = CLASSIFIERS[ARGS.classifier](ARGS)
    classifier.train()
