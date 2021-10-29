"""Filter iDigBio data."""
import re
from pathlib import Path
from sqlite3 import connect

import pandas as pd
from tqdm import tqdm
from .idigbio_load import MULTIMEDIA, OCCURRENCE, OCCURRENCE_RAW

# Sqlite3 and Python reserved words get renamed to include an "_" suffix
RESERVED = """ order class """

PHYLA = """ Anthophyta Angiospermae Magnoliophyta Magnolicae """.lower().split()
CLASSES = """ angiospermopsida dicotyledonae liliopsida magnoliopsida
    monocots monocotyledonae """.split()

pheno_terms = r""" \b (?: flower | fruit | petal | fls | corolla | leaves | tepal
            | seed | sterile | ray | infl | bract | inflor | inflorescence | stigma
            | sepal | flores ) \b """


def filter_records(in_db: Path, out_db: Path, chunk_size: int) -> None:
    """Remove records we can use from the data."""
    columns = sorted(set(MULTIMEDIA + OCCURRENCE + OCCURRENCE_RAW))

    renames = {c: c.split(":")[-1].lower() for c in columns}
    renames = {k: f'{v}_' if v in RESERVED else v for k, v in renames.items()}

    family_sql = """ select name from apg_ii_family_names
               union select name from apg_iv_family_names"""

    in_sql = """
        select *
        from multimedia
        join occurrence using (coreid)
        join occurrence_raw using (coreid)
    """

    with connect(in_db) as in_cxn, connect(out_db) as out_cxn:
        create_angiosperms_table(out_cxn, list(renames.values()))
        families = pd.read_sql(family_sql, in_cxn)

        reader = pd.read_sql(in_sql, in_cxn, index_col="coreid", chunksize=chunk_size)

        if_exists = "replace"

        for df in tqdm(reader):
            df = df.rename(columns=renames)

            df = df.loc[df["basisofrecord"] != "fossilspecimen"]

            keeps = df['phylum'].isin(PHYLA)
            keeps |= df['class_'].isin(CLASSES)
            keeps |= df['family'].isin(families['name'])
            df = df.loc[keeps]

            keeps = df['reproductivecondition'] != ''
            keeps |= df["occurrenceremarks"].str.contains(
                pheno_terms, flags=re.VERBOSE, case=False)
            keeps |= df["dynamicproperties"].str.contains(
                pheno_terms, flags=re.VERBOSE, case=False)
            keeps |= df["fieldnotes"].str.contains(
                pheno_terms, flags=re.VERBOSE, case=False)
            df = df.loc[keeps]

            df.to_sql("angiosperms", out_cxn, if_exists=if_exists, index=True)

            if_exists = "append"


def create_angiosperms_table(out_cxn, columns):
    """Create the angiosperm table."""
    fields = [f"{c} text" for c in columns if c != "coreid"]

    sql = f"""
        drop table if exists angiosperms;

        create table if not exists angiosperms (
            coreid integer primary key,
            {', '.join(fields)}
        );
    """
    out_cxn.executescript(sql)
