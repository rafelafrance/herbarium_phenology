#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from .pylib import consts
from .pylib import db_new
from .pylib import log
from .pylib import validate_args as val
from .pylib.models.all_models import MODELS
from .pylib.models.backbones import BACKBONES
from .pylib.runners import training_runner


def main():
    log.started()

    args = parse_args()
    orders = db_new.canned_select(args.database, "orders")

    model = MODELS[args.model](orders, args.backbone, args.load_model)

    training_runner.train(model, orders, args)

    log.finished()


def parse_args():
    description = """Train a model to classify herbarium traits."""
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
        help="""Which data split set to use.""",
    )

    arg_parser.add_argument(
        "--target-set",
        metavar="NAME",
        required=True,
        help="""Use this target set for trait target values.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=consts.TRAITS,
        required=True,
        help="""Train to classify this trait.""",
    )

    arg_parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys())[0],
        help="""Which model architecture.""",
    )

    arg_parser.add_argument(
        "--backbone",
        choices=list(BACKBONES.keys()),
        default=list(BACKBONES.keys())[0],
        help="""Which neural network backbone for the model.""",
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

    args = arg_parser.parse_args()

    val.validate_split_set(args.database, args.split_set)
    val.validate_target_set(args.database, args.target_set)

    return args


if __name__ == "__main__":
    main()
