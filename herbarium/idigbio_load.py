#!/usr/bin/env python
"""Load iDibBio data relevant to this project."""

import argparse
import textwrap
from pathlib import Path

import pylib.idigbio_load as idigbio_load
from pylib import log
from pylib.config import Config


def main(args):
    """Load the data."""
    if args.show_csv_files:
        print(idigbio_load.show_csv_files(args.zip_file))
        return

    if args.show_csv_headers:
        print(idigbio_load.show_csv_headers(args.zip_file, args.show_csv_headers))
        return

    idigbio_load.load_idigbio_data(args.database, args.zip_file, args.chunk_size)


def parse_args():
    """Process command-line arguments."""
    description = """
        Load iDigBio Data.

        The files in the iDigBio snapshot are too big to work with easily on a laptop.
        So, we extract CSV files from it and create a database from those CSV."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    configs = Config()
    defaults = configs.module_defaults()

    arg_parser.add_argument(
        "--database",
        default=defaults.idigbio_db,
        metavar="PATH",
        type=Path,
        help="""Path to the output SQLite3 database. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--zip-file",
        default=defaults.idigbio_zip_file,
        metavar="PATH",
        type=Path,
        help="""The zip file containing the iDigBio snapshot. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.idigbio_chunk,
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
    log.started()

    ARGS = parse_args()
    main(ARGS)

    log.finished()
