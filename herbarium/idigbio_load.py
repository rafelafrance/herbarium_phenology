#!/usr/bin/env python
"""Load iDibBio data relevant to this project."""
import argparse
import textwrap
from pathlib import Path

from pylib import log
from pylib.idigbio import idigbio_utils


def main():
    log.started()
    args = parse_args()

    if args.show_csv_files:
        print(idigbio_utils.show_csv_files(args.zip_file))
        return

    if args.show_csv_headers:
        print(idigbio_utils.show_csv_headers(args.zip_file, args.show_csv_headers))
        return

    idigbio_utils.load_idigbio_data(args.database, args.zip_file, args.chunk_size)

    log.finished()


def parse_args():
    description = """
        Load iDigBio Data.

        The files in the iDigBio snapshot are too big to work with easily on a laptop.
        So, we extract CSV files from it and create a database from those CSV."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--database",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the output SQLite3 database.""",
    )

    arg_parser.add_argument(
        "--zip-file",
        metavar="PATH",
        type=Path,
        required=True,
        help="""The zip file containing the iDigBio snapshot.""",
    )

    arg_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        metavar="N",
        help="""The number of lines read from the CSV file at a time.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--show-csv-files",
        action="store_true",
        help="""Show the list of CSV files in the iDigBio zip file and exit.""",
    )

    arg_parser.add_argument(
        "--show-csv-headers",
        metavar="NAME",
        help="""Show the of the headers of the iDigBio zip file's CSV and exit.
            Ex: --show-csv-headers occurrence.csv""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
