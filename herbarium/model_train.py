#!/usr/bin/env python3
"""Train a model to classify herbarium traits."""
import argparse
import sys
import textwrap
from pathlib import Path

from pylib import db
from pylib.herbarium_dataset import HerbariumDataset
from pylib.herbarium_model import BACKBONES
from pylib.herbarium_model import HerbariumModel
from pylib.herbarium_runner import HerbariumTrainingRunner


def parse_args():
    """Process command-line arguments."""
    description = """Train a herbarium phenology classifier model."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--database",
        "--db",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Path to the SQLite3 database (angiosperm data).""",
    )

    arg_parser.add_argument(
        "--save-model",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save best models to this path.""",
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
        "--backbone",
        choices=list(BACKBONES.keys()),
        default=list(BACKBONES.keys())[0],
        help="""Which neural network backbone to use.""",
    )

    arg_parser.add_argument(
        "--load-weights",
        type=Path,
        metavar="PATH",
        help="""Continue training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--log-dir",
        type=Path,
        metavar="PATH",
        help="""Output log files to this directory.""",
    )

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        metavar="FLOAT",
        default=0.001,
        help="""Initial learning rate. (default: %(default)s)""",
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
        "--epochs",
        type=int,
        metavar="INT",
        default=100,
        help="""How many epochs to train. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=HerbariumDataset.all_traits,
        default=HerbariumDataset.all_traits[0],
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

    trainer = HerbariumTrainingRunner(model, args.trait, orders, args)
    trainer.train()


if __name__ == "__main__":
    main()
