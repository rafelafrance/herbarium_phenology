"""Load iDigbBio data relevant to this project."""
import logging
import sqlite3
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd
from tqdm import tqdm

# The name of columns in the iDigBio zip file that we want
OCCURRENCE = """ coreid dwc:basisOfRecord dwc:kingdom dwc:phylum dwc:class
    dwc:order dwc:family dwc:genus dwc:specificEpithet dwc:scientificName
    dwc:eventDate dwc:continent dwc:country dwc:stateProvince dwc:county dwc:locality
    idigbio:geoPoint
    """.split()

# The name of columns we want in the iDigBio zip file
OCCURRENCE_RAW = """ coreid dwc:reproductiveCondition dwc:occurrenceRemarks
    dwc:dynamicProperties dwc:fieldNotes
    """.split()

# The name of columns we want in the iDigBio zip file
MULTIMEDIA = """ coreid accessURI """.split()

# Extra flags
FLAGS = """
        flowering      fruiting     leaf_out
    not_flowering  not_fruiting not_leaf_out """.split()


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


def _csv_reader(in_csv: IO[bytes], chunk_size: int, columns: list[str]):
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
