#!/usr/bin/env python
"""Load iDibBio data relevant to this project."""

from pathlib import Path
import argparse
import re
import sqlite3
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import tqdm

import pylib.log as log
from pylib.config import Configs


def load_data(args):
    """Load the data."""


def parse_args():
    """Process command-line arguments."""
    description = """
        Load iDigBio Data.

        The files in the iDigBio snapshot are too big to work with easily on a laptop.
        So, we extract one CSV file from them and then create a database table from
        that CSV."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    configs = Configs()
    defaults = configs.module_defaults()

    arg_parser.add_argument(
        "--database",
        default=defaults["database"],
        type=Path,
        help="""Path to the output SQLite3 database.""",
    )

    arg_parser.add_argument(
        "--zip-file",
        default=defaults["zip_file"],
        type=Path,
        help="""The zip file containing the iDigBio snapshot.""",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help="""The number of lines we read from the CSV file at a time.
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    log.started()

    ARGS = parse_args()
    load_data(ARGS)

    log.finished()
