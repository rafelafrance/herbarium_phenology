#!/usr/bin/env python3
"""Test a model that classifies herbarium traits."""
import argparse
import textwrap
from pathlib import Path

from models.backbone_params import BACKBONES
from models.herbarium_model import HerbariumModel
from pylib import db
from pylib import log
from pylib.const import TRAITS
from runners import inference_runner


def main():
    """Infer traits."""
    log.started()

    args = parse_args()
    orders = db.select_all_orders(args.database)

    model = HerbariumModel(orders, args.backbone, args.load_model)

    inference_runner.infer(model, orders, args)

    log.finished()


def parse_args():
    """Process command-line arguments."""
    description = """Run inference using a herbarium phenology trait classifier."""
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
        "--load-model",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Use this model for inference.""",
    )

    arg_parser.add_argument(
        "--inference-set",
        metavar="NAME",
        required=True,
        help="""Name this inference set.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=TRAITS,
        required=True,
        help="""Which trait to infer.""",
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
    return args


if __name__ == "__main__":
    main()
