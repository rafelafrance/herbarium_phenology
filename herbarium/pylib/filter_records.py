"""Filter iDigBio data."""
import sqlite3
from pathlib import Path

from tqdm import tqdm

from . import db
from . import idigbio_load as load
from . import pipeline

ON, OFF = 1, ""

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

# Search the row fields for flowering fruiting leaf_out in this order
SEARCH_FIELDS = """
    reproductivecondition occurrenceremarks fieldnotes dynamicproperties """.split()


def filter_records(in_db: Path, out_db: Path) -> None:
    """Remove records that are not angiosperms, have no phenological data, etc."""
    renames = get_column_renames()
    in_sql = build_select(renames)

    nlp = pipeline.pipeline()

    batch = []

    with sqlite3.connect(in_db) as in_cxn, sqlite3.connect(out_db) as out_cxn:
        in_cxn.row_factory = sqlite3.Row

        create_angiosperms_table(out_cxn, list(renames.values()) + load.FLAGS)

        for raw in tqdm(in_cxn.execute(in_sql)):
            row = dict(raw)

            # Add empty flags to the record
            for field in load.FLAGS:
                row[field] = OFF

            for field in SEARCH_FIELDS:
                if set_row_flags(row, nlp, field):
                    batch.append(row)
                    break

    db.insert_batch(out_db, build_insert(renames), batch)


def set_row_flags(row, nlp, field):
    """Set the row flags for traits."""
    flags_set = False
    if not row[field]:
        return flags_set

    doc = nlp(row[field])
    for ent in doc.ents:
        trait = ent._.data["trait"]

        flags = REPRO_LABELS if field == "reproductivecondition" else LABELS

        for flag in flags.get(trait, []):
            row[flag] = ON
            flags_set = True

    return flags_set


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
            having count(*) > 1),
          families as (select name as family from apg_ii_family_names
                 union select name           from apg_iv_family_names)
        select {fields}
        from multimedia
        join occurrence using (coreid)
        join occurrence_raw using (coreid)
       where coreid not in (select coreid from multiples)
         and accessuri <> ''
         and `dwc:basisofrecord` <> 'fossilspecimen'
         and (   `dwc:phylum` in (select phylum from phyla)
              or `dwc:class`  in (select class_ from classes)
              or `dwc:order`  in (select order_ from orders)
              or `dwc:family` in (select family from families)
             );
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
