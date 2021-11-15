"""Filter iDigBio data."""

import sqlite3
from pathlib import Path
from sqlite3 import connect

from tqdm import tqdm

from . import db
from . import idigbio_load as load
from . import pipeline


LABELS = {
    "flowering": ["flowering"],
    "flowering_fruiting": ["flowering", "fruiting"],
    "fruiting": ["fruiting"],
    "leaf_out": ["leaf_out"],
    "not_flowering": ["not_flowering"],
    "not_fruiting": ["not_fruiting"],
    "not_leaf_out": ["not_leaf_out"],
}

REPRO_LABELS = LABELS | {
    "abbrev_flowering": ["flowering"],
    "abbrev_flowering_fruiting": ["flowering", "fruiting"],
    "abbrev_fruiting": ["fruiting"],
    "abbrev_leaf_out": ["leaf_out"],
}

ON = 1
OFF = ""


def filter_records(in_db: Path, out_db: Path) -> None:
    """Remove records that are not angiosperms, have no phenological data, etc."""
    renames = get_column_renames()
    in_sql = build_select(renames)

    nlp = pipeline.pipeline()

    batch = []

    with connect(in_db) as in_cxn, connect(out_db) as out_cxn:
        in_cxn.row_factory = sqlite3.Row

        phyla = taxa(in_cxn, "select * from phyla")
        classes = taxa(in_cxn, "select * from classes")
        orders = taxa(in_cxn, "select * from orders")
        families = get_families(in_cxn)

        create_angiosperms_table(out_cxn, list(renames.values()) + load.FLAGS)

        for raw in tqdm(in_cxn.execute(in_sql)):
            row = dict(raw)

            if is_fossil(row):
                continue

            if not is_angiosperm(row, phyla, classes, orders, families):
                continue

            # Add empty flags to the record
            for field in load.FLAGS:
                row[field] = OFF

            set_row_flags(row, nlp, "reproductivecondition")
            set_row_flags(row, nlp, "dynamicproperties")
            set_row_flags(row, nlp, "occurrenceremarks")
            set_row_flags(row, nlp, "fieldnotes")

            if any(row[f] for f in load.FLAGS):
                batch.append(row)

    db.insert_batch(out_db, build_insert(renames), batch)


def set_row_flags(row, nlp, field):
    """Set the row flags for traits."""
    doc = nlp(row[field])
    for ent in doc.ents:
        trait = ent._.data["trait"]
        flags = REPRO_LABELS if field == "reproductivecondition" else LABELS
        for flag in flags.get(trait, []):
            row[flag] = ON


def get_families(in_cxn):
    """get the family names for angiosperms."""
    family_sql = """
              select name from apg_ii_family_names
        union select name from apg_iv_family_names
        """
    families = taxa(in_cxn, family_sql)
    return families


def get_column_renames():
    """Get the output column names."""
    columns = sorted(set(load.MULTIMEDIA + load.OCCURRENCE + load.OCCURRENCE_RAW))
    renames = {c: c.split(":")[-1].lower() for c in columns}
    renames["dwc:class"] = "class_"
    renames["dwc:order"] = "order_"
    return renames


def build_select(renames):
    """Build the select statement for reading iDigBio records."""
    columns = [f"`{k}` as {v}" for k, v in renames.items()]
    fields = ", ".join(columns)
    sql = f"""
          with multiples as (
            select coreid
              from multimedia
          group by coreid
            having count(*) > 1)
        select {fields}
        from multimedia
        join occurrence using (coreid)
        join occurrence_raw using (coreid)
       where coreid not in (select coreid from multiples)
         and accessuri <> ''
       """
    return sql


def build_insert(renames):
    """Build an insert statement for the angiosperm records."""
    columns = list(renames.values()) + load.FLAGS
    fields = ", ".join(columns)
    values = [f":{f}" for f in columns]
    values = ", ".join(values)
    sql = f""" insert into angiosperms ({fields}) values ({values}); """
    return sql


def is_fossil(row):
    """Remove fossils from the data."""
    return row["basisofrecord"] == "fossilspecimen"


def is_angiosperm(row, phyla, classes, orders, families):
    """Only keep angiosperm records."""
    return (row["phylum"] in phyla or row["class_"] in classes
            or row["order_"] in orders or row["family"] in families)


def taxa(in_cxn: sqlite3.Connection, sql: str):
    """Get taxon names."""
    return [r[0] for r in in_cxn.execute(sql)]


def create_angiosperms_table(out_cxn, columns):
    """Create the angiosperm table."""
    fields = [f"            {c} text" for c in columns if c != "coreid"]
    fields = ",\n".join(fields)

    sql = f"""
        drop table if exists angiosperms;

        create table if not exists angiosperms (
            coreid text primary key,
            {fields}
        );
    """
    out_cxn.executescript(sql)
