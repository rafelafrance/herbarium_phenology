#!/usr/bin/env python
"""Load iDibBio data relevant to this project."""
import argparse
import logging
import sqlite3
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from herbarium.pylib import log
from herbarium.pylib.download_images import MULTIMEDIA
from herbarium.pylib.download_images import OCCURRENCE
from herbarium.pylib.download_images import OCCURRENCE_RAW


def load_idigbio_data(db: Path, zip_file: Path, chunk_size: int) -> None:
    """Load the iDigBio data into a database."""
    coreid = load_multimedia(db, zip_file, chunk_size)
    load_csv(db, zip_file, chunk_size, "occurrence", OCCURRENCE, coreid)
    load_csv(db, zip_file, chunk_size, "occurrence_raw", OCCURRENCE_RAW, coreid)


def load_multimedia(db: Path, zip_file: Path, chunk_size: int) -> set[str]:
    """Load the multimedia.csv file into the sqlite3 database."""
    table = "multimedia"

    logging.info(f"Loading {table}")

    coreid = set()

    with sqlite3.connect(db) as cxn:
        with zipfile.ZipFile(zip_file) as zippy:
            with zippy.open(f"{table}.csv") as in_csv:
                reader = _csv_reader(in_csv, chunk_size, MULTIMEDIA)

                if_exists = "replace"

                for df in tqdm(reader):
                    coreid |= set(df.index.tolist())

                    df.to_sql(table, cxn, if_exists=if_exists)
                    if_exists = "append"
    return coreid


def load_csv(
    db: Path,
    zip_file: Path,
    chunk_size: int,
    table: str,
    columns: list[str],
    coreid: set[str],
) -> None:
    """Load an occurrence*.csv file into the sqlite3 database."""
    logging.info(f"Loading {table}")

    with sqlite3.connect(db) as cxn:
        with zipfile.ZipFile(zip_file) as zippy:
            with zippy.open(f"{table}.csv") as in_csv:
                reader = _csv_reader(in_csv, chunk_size, columns)

                if_exists = "replace"

                for df in tqdm(reader):
                    df = df.loc[df.index.isin(coreid)]

                    df.to_sql(table, cxn, if_exists=if_exists)

                    if_exists = "append"


def _csv_reader(in_csv: Any, chunk_size: int, columns: list[str]):
    return pd.read_csv(
        in_csv,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        index_col="coreid",
        usecols=list(columns),
    )


def show_csv_headers(zip_file: Path, csv_file: str) -> list[str]:
    """Get the headers of the given CSV file within the iDigBio zip file."""
    with zipfile.ZipFile(zip_file) as zippy:
        with zippy.open(csv_file) as in_file:
            header = in_file.readline()
    headers = [h.decode().strip() for h in sorted(header.split(b","))]
    return headers


def show_csv_files(zip_file: Path) -> list[str]:
    """Get the list of CSV files within the iDigBio zip file."""
    with zipfile.ZipFile(zip_file) as zippy:
        return zippy.namelist()


def parse_args():
    """Process command-line arguments."""
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


def main():
    """Load the data."""
    log.started()
    args = parse_args()

    if args.show_csv_files:
        print(show_csv_files(args.zip_file))
        return

    if args.show_csv_headers:
        print(show_csv_headers(args.zip_file, args.show_csv_headers))
        return

    load_idigbio_data(args.database, args.zip_file, args.chunk_size)

    log.finished()


if __name__ == "__main__":
    main()
