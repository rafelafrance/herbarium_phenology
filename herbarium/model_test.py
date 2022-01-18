#!/usr/bin/env python3
"""Test a model that classifies herbarium traits."""
import argparse
import sys
import textwrap
from pathlib import Path

from pylib import db
from pylib.herbarium_dataset import ALL_TRAITS
from pylib.herbarium_model import BACKBONES
from pylib.herbarium_model import HerbariumModel
from pylib.herbarium_runner import HerbariumTestingRunner


def parse_args():
    """Process command-line arguments."""
    description = """Test a herbarium phenology classifier model."""
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
        "--backbone",
        choices=list(BACKBONES.keys()),
        default=list(BACKBONES.keys())[0],
        help="""Which neural network backbone to use.""",
    )

    arg_parser.add_argument(
        "--load-weights",
        type=Path,
        required=True,
        help="""Use this model for testing.""",
    )

    arg_parser.add_argument(
        "--test-run",
        metavar="NAME",
        required=True,
        help="""Name this test run. Test results are stored in the database.""",
    )

    arg_parser.add_argument(
        "--split-run",
        metavar="NAME",
        required=True,
        help="""Which data split to use. Splits are saved in the database and each
            one is used for a specific purpose. So, the split-run must be in the
            database.""",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        metavar="INT",
        default=16,
        help="""Input batch size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--workers",
        type=int,
        metavar="INT",
        default=4,
        help="""Number of workers for loading data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--trait",
        nargs="*",
        choices=ALL_TRAITS,
        default=ALL_TRAITS[0],
        help="""Which trait to classify.""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()

    split_runs = db.select_all_split_runs(args.database)
    if args.split_run not in split_runs:
        print(f"{args.split_run} is not in split_runs. Valid split_runs:")
        for run in split_runs:
            print(run)
        sys.exit(1)

    return args


def main():
    """Train a model using just pytorch."""
    args = parse_args()
    orders = db.select_orders(args.database, args.split_run)

    model = HerbariumModel(orders, args)

    runner = HerbariumTestingRunner(model, args.trait, orders, args)
    runner.test()


if __name__ == "__main__":
    main()
