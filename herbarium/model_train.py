#!/usr/bin/env python3
"""Train a model to classify herbarium traits."""
import argparse
import textwrap
from pathlib import Path

from herbarium.models.backbone_params import BACKBONES
from herbarium.models.herbarium_full_model import HerbariumFullModel
from herbarium.models.herbarium_model import HerbariumModel
from herbarium.pylib import const
from herbarium.pylib import db
from herbarium.pylib import log
from herbarium.pylib import validate_args as val
from herbarium.runners import training_runner

# from pylib.herbarium_model_exp import HydraModel


def main():
    """Train a model using just pytorch."""
    log.started()

    args = parse_args()
    orders = db.select_all_orders(args.database)

    if args.experiment:
        model = HerbariumFullModel(orders, args.backbone, args.load_model)
        # model = HydraModel(orders, args.backbone, args.load_model, args.trait)
    else:
        model = HerbariumModel(orders, args.backbone, args.load_model)

    training_runner.train(model, orders, args)

    log.finished()


def parse_args():
    """Process command-line arguments."""
    description = """Train a herbarium phenology trait classifier."""
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
        help="""Use this target set for trait target values.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=const.TRAITS,
        required=True,
        help="""Train to classify this trait.""",
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
        "--experiment",
        action="store_true",
        help="""Run an experimental model.""",
    )
    args = arg_parser.parse_args()

    val.validate_split_set(args.database, args.split_set)
    val.validate_target_set(args.database, args.target_set)

    return args


if __name__ == "__main__":
    main()
