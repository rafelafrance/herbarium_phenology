#!/usr/bin/env python
"""Filter iDigBio records."""
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

from pylib import log
from pylib.filter_records import filter_records


def main(args: argparse.Namespace) -> None:
    """Filter the records."""
    filter_records(args.in_db, args.out_db, args.filter_run)


def parse_args() -> argparse.Namespace:
    """Process command-line arguments."""
    description = """Filter records to only include angiosperm records with
        a single image and it must contain a notation on flowering, fruiting, or
        leaf-out."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--in-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the input SQLite3 database (iDigBio data).""",
    )

    arg_parser.add_argument(
        "--out-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the output SQLite3 database (angiosperm data).""",
    )

    default = datetime.now().isoformat(sep="_", timespec="seconds")
    arg_parser.add_argument(
        "--filter-run",
        default=default,
        help="""Name the filter run to allow multiple extracts.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    log.started()

    ARGS = parse_args()
    main(ARGS)

    log.finished()
