#!/usr/bin/env python3
"""Test a model that classifies herbarium traits."""
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib.herbarium_dataset import HerbariumDataset
from pylib.multi_efficient_net import NETS
from pylib.train_model import test


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
        "--net",
        choices=list(NETS.keys()),
        default=list(NETS.keys())[0],
        help="""Which neural network to use.""",
    )

    arg_parser.add_argument(
        "--load-weights",
        type=Path,
        required=True,
        help="""Use this model.""",
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
        "--split-run",
        default="first_split",
        help="""Which data split to use. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--trait",
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


if __name__ == "__main__":
    ARGS = parse_args()
    ORDERS = db.select_orders(ARGS.database, ARGS.split_run)
    NET = NETS[ARGS.net](len(ORDERS), ARGS.load_weights, freeze="all")
    test(ARGS, NET, ORDERS)
