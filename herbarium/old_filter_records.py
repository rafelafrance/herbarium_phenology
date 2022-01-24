#!/usr/bin/env python
"""Filter iDigBio records."""
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

from pylib import filter_records as fr
from pylib import log


def parse_args() -> argparse.Namespace:
    """Process command-line arguments."""
    description = """TODO: Rewrite this to allow multiple NLP runs and records without
        notations."""

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
        "--filter-set",
        default=default,
        help="""Name the filter set to allow multiple extracts.""",
    )

    args = arg_parser.parse_args()
    return args


def main() -> None:
    """Filter the records."""
    log.started()
    args = parse_args()
    fr.filter_records(args.in_db, args.out_db, args.filter_set)
    log.finished()


if __name__ == "__main__":
    main()
