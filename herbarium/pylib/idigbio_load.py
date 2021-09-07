#!/usr/bin/env python
"""Load iDibBio data relevant to this project."""

import logging
import sqlite3
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd
import tqdm

# The name of columns in the iDigBio zip file & what they're renamed to in the database
OCCURRENCE: dict[str, str] = {
    # original CSV field name -> database column name
    "coreid": "coreid",
    "dwc:basisOfRecord": "basisOfRecord",
    "dwc:order": "order_",
    "dwc:family": "family",
    "dwc:genus": "genus",
    "dwc:specificEpithet": "specificEpithet",
    "dwc:scientificName": "scientificName",
    "dwc:eventDate": "eventDate",
    "dwc:continent": "continent",
    "dwc:country": "country",
    "dwc:stateProvince": "stateProvince",
    "dwc:county": "county",
    "dwc:locality": "locality",
    "idigbio:geoPoint": "geoPoint",
}

# The name of columns in the iDigBio zip file & what they're renamed to in the database
OCCURRENCE_RAW: dict[str, str] = {
    # original CSV field name -> database column name
    "coreid": "coreid",
    "dwc:reproductiveCondition": "reproductiveCondition",
    "dwc:occurrenceRemarks": "occurrenceRemarks",
    "dwc:dynamicProperties": "dynamicProperties",
    "dwc:fieldNotes": "fieldNotes",
}

# The name of columns in the iDigBio zip file & what they're renamed to in the database
MULTIMEDIA: dict[str, str] = {
    # original CSV field name -> database column name
    "coreid": "coreid",
    "ac:accessURI": "accessURI",
}


def load_idigbio_data(db: Path, zip_file: Path, chunk_size: int) -> None:
    """Load the iDigBio data into a database.

    Args:
        db:         path to the output SQLite3 database
        zip_file:   path to the zip file containing the input iDigBio snapshot
        chunk_size: how many CSV records to read at a time

    Returns:
        None
    """
    coreid = load_multimedia(db, zip_file, chunk_size)
    load_csv(db, zip_file, chunk_size, "occurrence", OCCURRENCE, coreid)
    load_csv(db, zip_file, chunk_size, "occurrence_raw", OCCURRENCE_RAW, coreid)


def load_multimedia(db: Path, zip_file: Path, chunk_size: int) -> set[str]:
    """Load the multimedia.csv file into the sqlite3 database.

    Args:
        db:         path to the output SQLite3 database
        zip_file:   path to the zip file containing the input iDigBio snapshot
        chunk_size: how many CSV records to read at a time

    Returns:
        A set of all core IDs within the multimedia.csv file
    """
    table = "multimedia"
    columns = MULTIMEDIA

    logging.info(f"Loading {table}")

    coreid = set()

    with sqlite3.connect(db) as cxn:
        with zipfile.ZipFile(zip_file) as zippy:
            with zippy.open(f"{table}.csv") as in_csv:
                reader = _csv_reader(in_csv, chunk_size, columns)

                if_exists = "replace"

                for df in tqdm.tqdm(reader):
                    df = df.rename(columns=columns)

                    coreid |= set(df.index.tolist())

                    df.to_sql(table, cxn, if_exists=if_exists)
                    if_exists = "append"
    return coreid


def load_csv(
        db: Path,
        zip_file: Path,
        chunk_size: int,
        table: str,
        columns: dict[str, str],
        coreid: set[str],
) -> None:
    """Load an occurrence*.csv file into the sqlite3 database.

    Args:
        db:         path to the output SQLite3 database
        zip_file:   path to the zip file containing the input iDigBio snapshot
        chunk_size: how many CSV records to read at a time
        table:      which table/CSV file are we reading or writing to
        columns:    the mapping from CSV column names to SQLite3 column names
        coreid:     which core IDs to keep. This filters out records without media

    Returns:
        None
    """
    logging.info(f"Loading {table}")

    with sqlite3.connect(db) as cxn:
        with zipfile.ZipFile(zip_file) as zippy:
            with zippy.open(f"{table}.csv") as in_csv:
                reader = _csv_reader(in_csv, chunk_size, columns)

                if_exists = "replace"

                for df in tqdm.tqdm(reader):
                    df = df.rename(columns=columns)

                    df = df.loc[df.index.isin(coreid)]

                    if "basisOfRecord" in df.columns:
                        df = df.loc[df["basisOfRecord"] != "fossilspecimen"]
                    if "genus" in df.columns:
                        df = df.loc[df["genus"] != ""]

                    df.to_sql(table, cxn, if_exists=if_exists)
                    if_exists = "append"


def _csv_reader(in_csv: IO[bytes], chunk_size: int, columns: dict[str, str]):
    """Get a pandas CSV reader.

    Args:
        in_csv:     an opened file handle to the CSV file withing the iDibBio zip file
        chunk_size: how many CSV records to read at a time
        columns:    limit the columns read to the keys in this dict

    Returns:
        pandas TextFileReader
    """
    return pd.read_csv(
        in_csv,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        index_col="coreid",
        usecols=list(columns.keys()),
    )


def show_csv_headers(zip_file: Path, csv_file: str) -> list[str]:
    """Get the headers of the given CSV file within the iDigBio zip file.

    Args:
        zip_file: path to the zip file containing the input iDigBio snapshot
        csv_file: the name of a CSV file within the iDigBio snapshot

    Returns:
        A list of CSV column headers
    """
    with zipfile.ZipFile(zip_file) as zippy:
        with zippy.open(csv_file) as in_file:
            header = in_file.readline()
    headers = [h.decode().strip() for h in sorted(header.split(b','))]
    return headers


def show_csv_files(zip_file: Path) -> list[str]:
    """Get the list of CSV files within the iDigBio zip file.

    Args:
        zip_file: path to the zip file containing the input iDigBio snapshot

    Returns:
        A list of CSV files in the iDigBio snapshot
    """
    with zipfile.ZipFile(zip_file) as zippy:
        return zippy.namelist()
