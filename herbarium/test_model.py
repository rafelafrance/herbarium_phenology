#!/usr/bin/env python3
"""Test a model that classifies herbarium traits."""
import argparse
import textwrap
from pathlib import Path

from pylib.classifier import CLASSIFIERS


def parse_args():
    """Process command-line arguments."""
    description = """Test a herbarium phenology classifier."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--database",
        "--db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the SQLite3 database (angiosperm data).""",
    )

    arg_parser.add_argument(
        "--classifier",
        choices=list(CLASSIFIERS.keys()),
        default=list(CLASSIFIERS.keys())[0],
        help="""Which classifier model to use.""",
    )

    arg_parser.add_argument("--prev-model", required=True, help="""Use this model.""")

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="""Input batch size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="""Number of workers for loading data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--split-run",
        default="first_split",
        help="""Which data split to use. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    classifier = CLASSIFIERS[ARGS.classifier](ARGS)
    classifier.train()
