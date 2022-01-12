#!/usr/bin/env python3
"""Train a model to classify herbarium traits."""
import argparse
import sys
import textwrap
from pathlib import Path

from pylib import db
from pylib.efficient_net import BACKBONES
from pylib.efficient_net import EfficientNet
from pylib.herbarium_dataset import HerbariumDataset
from pylib.run_model import train


def parse_args():
    """Process command-line arguments."""
    description = """Train a herbarium phenology classifier model."""
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
        "--save-model", required=True, help="""Save best models to this path."""
    )

    arg_parser.add_argument(
        "--split-run", required=True, help="""Which data split to use."""
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
        help="""Start training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=0.001,
        help="""Initial learning rate. (default: %(default)s)""",
    )

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
        "--epochs",
        type=int,
        default=100,
        help="""How many epochs to train. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--freeze",
        choices=["top", "all"],
        default="top",
        help="""Freeze model layers.""",
    )

    arg_parser.add_argument(
        "--trait",
        nargs="*",
        choices=HerbariumDataset.all_traits,
        default=HerbariumDataset.all_traits[0],
        help="""Which trait to classify. You may use this argument multiple times.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
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
    net = EfficientNet(args.backbone, orders, args.load_weights)
    train(args, net, orders)


if __name__ == "__main__":
    main()
