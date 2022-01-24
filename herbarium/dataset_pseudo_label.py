#!/usr/bin/env python3
"""Create split runs from inferred traits."""
import argparse
import textwrap
from pathlib import Path


def parse_args():
    """Process command-line arguments."""
    description = """Run inference using a herbarium phenology classifier model and use
        the inferences to create new pseudo-traits for the images."""
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
        "--split-set",
        metavar="NAME",
        required=True,
        help="""Which data split to create. Splits are saved in the database and each
            one is used for a specific purpose.""",
    )

    arg_parser.add_argument(
        "--inference-set",
        metavar="NAME",
        required=True,
        help="""Name this inference set. Inference results are stored in the
            database.""",
    )

    arg_parser.add_argument(
        "--min-threshold",
        type=float,
        metavar="FLOAT",
        default=0.1,
        help="""When the inference value for the trait is below this the trait is
            considered to be absent. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-threshold",
        type=float,
        metavar="FLOAT",
        default=0.9,
        help="""When the inference value for the trait is above this the trait is
            considered to be present. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--train-split",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the validation. I.e. evaluating
            training progress at the end of each epoch. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the testing. I.e. the holdout
            data used to evaluate the model after training. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()
    return args


def main():
    """Infer traits."""
    # args = parse_args()


if __name__ == "__main__":
    main()
