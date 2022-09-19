#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib import split_utils
from pylib import validate_args as val
from pylib.consts import TRAITS


def main():
    args = parse_args()
    orders = db.canned_select(args.database, "orders")
    split_utils.assign_records(args, orders)


def parse_args():
    description = """Split labeled images into training, testing, and validation
        datasets. Note: We attempt to spread images from each order into all three
        datasets."""
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
        help="""Give the split set this name.""",
    )

    arg_parser.add_argument(
        "--target-set",
        metavar="NAME",
        required=True,
        help="""Use this target set for target values.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=TRAITS,
        required=True,
        help="""Which trait to classify.""",
    )

    arg_parser.add_argument(
        "--base-split-set",
        metavar="NAME",
        help="""Start with this split set. Add split records from this set before
            adding any new records from the --target-set. The is so we don't train
            on a previous test data when updating a model with new data.""",
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

    args = arg_parser.parse_args()

    val.validate_target_set(args.database, args.target_set)
    if args.base_split_set:
        val.validate_split_set(args.database, args.base_split_set)

    return args


if __name__ == "__main__":
    main()
