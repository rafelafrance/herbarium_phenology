#!/usr/bin/env python3
"""Create split runs from images."""
import argparse
import textwrap
from pathlib import Path

from pylib import dataset_split as ds
from pylib.const import ALL_TRAITS


def parse_args():
    """Process command-line arguments."""
    description = """Split images into training, validation, and test sets."""
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
        "--split-set",
        metavar="NAME",
        required=True,
        help="""Which data split to create. Splits are saved in the database and each
            one is used for a specific purpose.""",
    )

    arg_parser.add_argument(
        "--trait",
        nargs="*",
        choices=ALL_TRAITS,
        help="""Filter the data so that the dataset contains this trait. You may use
            this argument more than once.""",
    )

    arg_parser.add_argument(
        "--train-split",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the validation. I.e. evaluating
            training progress at the end of each epoch. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the testing. I.e. the holdout
            data used to evaluate the model after training. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()
    return args


def main():
    """Infer traits."""
    args = parse_args()
    if not args.trait:
        args.trait = ALL_TRAITS

    ds.assign_records(args)


if __name__ == "__main__":
    main()
