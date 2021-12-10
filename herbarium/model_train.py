#!/usr/bin/env python3
"""Train a model to classify herbarium traits."""
import argparse
import textwrap
from pathlib import Path

from pylib.net import NETS
from pylib.train_model import train


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
        "--net",
        choices=list(NETS.keys()),
        default=list(NETS.keys())[0],
        help="""Which neural network to use.""",
    )

    arg_parser.add_argument("--prev-model", help="""Use this model.""")

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=0.0005,
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
        "--freeze", action="store_true", help="""Freeze the model top."""
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
    NET = NETS[ARGS.net](ARGS)
    train(ARGS, NET)