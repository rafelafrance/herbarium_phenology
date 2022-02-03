#!/usr/bin/env python3
"""Create a target dataset from inferred traits."""
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib import log
from pylib import validate_args as val
from pylib.const import ALL_TRAITS
from pylib.herbarium_model import BACKBONES
from pylib.herbarium_model import HerbariumModel
from pylib.herbarium_runner import HerbariumPseudoRunner


def parse_args():
    """Process command-line arguments."""
    description = """Create a target dataset from an inference set."""
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
        "--save-model",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save best models to this path.""",
    )

    arg_parser.add_argument(
        "--split-set",
        metavar="NAME",
        required=True,
        help="""Which data split to use. Splits are saved in the database and each
            one is used for a specific purpose.""",
    )

    arg_parser.add_argument(
        "--target-set",
        metavar="NAME",
        required=True,
        help="""Give the target dataset this name.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=ALL_TRAITS,
        required=True,
        help="""Which trait to infer.""",
    )

    arg_parser.add_argument(
        "--backbone",
        choices=list(BACKBONES.keys()),
        default=list(BACKBONES.keys())[0],
        help="""Which neural network backbone to use.""",
    )

    arg_parser.add_argument(
        "--load-model",
        type=Path,
        metavar="PATH",
        help="""Continue training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        help="""Save tensorboard logs to this directory.""",
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
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    arg_parser.add_argument(
        "--unlabeled-limit",
        type=int,
        metavar="INT",
        help="""How many unlabeled images to use.""",
    )

    arg_parser.add_argument(
        "--pseudo-max",
        type=float,
        metavar="FLOAT",
        default=3.0,
        help="""Final pseudo label loss weight. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--pseudo-start",
        type=float,
        metavar="INT",
        default=0,
        help="""Start adding pseudo labels at this epoch.""",
    )

    args = arg_parser.parse_args()

    val.validate_split_set(args.database, args.split_set)
    val.validate_target_set(args.database, args.target_set)

    return args


def main():
    """Infer traits."""
    log.started()

    args = parse_args()
    orders = db.select_all_orders(args.database)

    model = HerbariumModel(orders, args.backbone, args.load_model)

    runner = HerbariumPseudoRunner(model, orders, args)
    runner.run()

    log.finished()


if __name__ == "__main__":
    main()
