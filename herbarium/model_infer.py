#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib import log
from pylib.consts import TRAITS
from pylib.engines import inference_engine
from pylib.models.all_models import MODELS
from pylib.models.backbones import BACKBONES


def main():
    log.started()

    args = parse_args()
    orders = db.canned_select(args.database, "orders")

    model = MODELS[args.model](orders, args.backbone, args.load_model)

    inference_engine.infer(model, orders, args)

    log.finished()


def parse_args():
    description = """Infer herbarium traits with a trained model."""
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
