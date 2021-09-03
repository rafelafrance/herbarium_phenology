#!/usr/bin/env python
"""Load iDibBio data relevant to this project."""

import logging
import sqlite3
import zipfile

import pandas as pd
import tqdm

OCCURRENCE = {
    # original field name -> database column name
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

OCCURRENCE_RAW = {
    # original field name -> database column name
    "coreid": "coreid",
    "dwc:reproductiveCondition": "reproductiveCondition",
    "dwc:occurrenceRemarks": "occurrenceRemarks",
    "dwc:dynamicProperties": "dynamicProperties",
    "dwc:fieldNotes": "fieldNotes",
}

MULTIMEDIA = {
    # original field name -> database column name
    "coreid": "coreid",
    "ac:accessURI": "accessURI",
}


def load_idigbio_data(db, zip_file, chunk_size):
    """Load the iDigBio data into a database."""
    coreid = load_multimedia(db, zip_file, chunk_size, "multimedia", MULTIMEDIA)
    load_csv(db, zip_file, chunk_size, "occurrence", OCCURRENCE, coreid)
    load_csv(db, zip_file, chunk_size, "occurrence_raw", OCCURRENCE_RAW, coreid)


def load_multimedia(db, zip_file, chunk_size, table, columns):
    """Load a CSV file into the database."""
    logging.info(f"Loading {table}")
    coreid = set()

    with sqlite3.connect(db) as cxn:
        with zipfile.ZipFile(zip_file) as zippy:
            with zippy.open(f"{table}.csv") as in_csv:
                reader = csv_reader(in_csv, chunk_size, columns)

                if_exists = "replace"

                for df in tqdm.tqdm(reader):
                    df = df.rename(columns=columns)

                    coreid |= set(df.index.tolist())

                    df.to_sql(table, cxn, if_exists=if_exists)
                    if_exists = "append"
    return coreid


def load_csv(db, zip_file, chunk_size, table, columns, coreid):
    """Load a CSV file into the database."""
    logging.info(f"Loading {table}")

    with sqlite3.connect(db) as cxn:
        with zipfile.ZipFile(zip_file) as zippy:
            with zippy.open(f"{table}.csv") as in_csv:
                reader = csv_reader(in_csv, chunk_size, columns)

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


def csv_reader(in_csv, chunk_size, columns):
    """Get a pandas CSV reader."""
    return pd.read_csv(
        in_csv,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        index_col="coreid",
        usecols=list(columns.keys()),
    )


def show_csv_headers(zip_file, csv_file):
    """Show the headers of the given CSV file in the zip file and exit."""
    with zipfile.ZipFile(zip_file) as zippy:
        with zippy.open(csv_file) as in_file:
            header = in_file.readline()
    headers = [h.decode().strip() for h in sorted(header.split(b','))]
    return headers


def show_csv_files(zip_file):
    """Show the list of CSV files in the iDigBio zip file."""
    with zipfile.ZipFile(zip_file) as zippy:
        return zippy.namelist()
