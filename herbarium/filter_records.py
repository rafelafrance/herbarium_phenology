#!/usr/bin/env python
"""Filter iDigBio records."""
import argparse
import textwrap
from pathlib import Path

from pylib import log
from pylib.filter_records import filter_records


def main(args: argparse.Namespace) -> None:
    """Filter the records."""
    filter_records(args.in_db, args.out_db)


def parse_args() -> argparse.Namespace:
    """Process command-line arguments."""
    description = """Filter records."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--in-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the input SQLite3 database.""",
    )

    arg_parser.add_argument(
        "--out-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the output SQLite3 database.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    log.started()

    ARGS = parse_args()
    main(ARGS)

    log.finished()
