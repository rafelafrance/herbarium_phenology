#!/usr/bin/env python3
"""Test a model that classifies herbarium traits."""
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib.efficient_net import BACKBONES
from pylib.efficient_net import EfficientNet
from pylib.herbarium_dataset import HerbariumDataset
from pylib.run_model import infer


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
        help="""Which neural network to use.""",
    )

    arg_parser.add_argument(
        "--load-weights",
        type=Path,
        required=True,
        help="""Use this model.""",
    )

    arg_parser.add_argument(
        "--inference-run",
        required=True,
        help="""Name this inference run.""",
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
        "--trait",
        nargs="*",
        choices=HerbariumDataset.all_traits,
        default=HerbariumDataset.all_traits[0],
        help="""Which trait to classify. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()
    return args


def main():
    """Infer traits."""
    args = parse_args()
    orders = db.select_orders(args.database, args.split_run)
    net = EfficientNet(
        args.backbone, orders, args.load_weights, args.freeze, args.trait
    )
    infer(args, net, orders)


if __name__ == "__main__":
    main()
