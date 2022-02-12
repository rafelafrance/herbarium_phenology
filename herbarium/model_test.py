#!/usr/bin/env python3
"""Test a model that classifies utils traits."""
import argparse
import textwrap
from pathlib import Path

from pylib.models.all_models import MODELS
from pylib.models.backbones import BACKBONES
from pylib.runners import testing_runner
from pylib.utils import db
from pylib.utils import log
from pylib.utils import validate_args as val
from pylib.utils.const import TRAITS


def main():
    """Train a model using just pytorch."""
    log.started()

    args = parse_args()
    orders = db.select_all_orders(args.database)

    model = MODELS[args.model](orders, args.backbone, args.load_model)

    testing_runner.test(model, orders, args)

    log.finished()


def parse_args():
    """Process command-line arguments."""
    description = """Test a utils utils trait classifier."""
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
        required=True,
        help="""Use this model for testing.""",
    )

    arg_parser.add_argument(
        "--test-set",
        metavar="NAME",
        required=True,
        help="""Name this test set.""",
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
        choices=TRAITS,
        required=True,
        help="""Which trait to classify.""",
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
