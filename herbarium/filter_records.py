# #!/usr/bin/env python
# """Filter iDigBio records."""
# import argparse
# import sqlite3
# import textwrap
# from datetime import datetime
# from pathlib import Path
#
# from pylib import db
# from pylib import log
# from pylib import pipeline
# from tqdm import tqdm
#
# from junk import idigbio_load as load
#
# ON, OFF = 1, ""
#
# LABELS = {
#     "flowering": ["flowering"],
#     "flowering_fruiting": ["flowering", "fruiting"],
#     "fruiting": ["fruiting"],
#     "leaf_out": ["leaf_out"],
#     "not_flowering": ["not_flowering"],
#     "not_fruiting": ["not_fruiting"],
#     "not_leaf_out": ["not_leaf_out"],
# }
#
# REPRO_LABELS = LABELS | {
#     "abbrev_flowering": ["flowering"],
#     "abbrev_flowering_fruiting": ["flowering", "fruiting"],
#     "abbrev_fruiting": ["fruiting"],
#     "abbrev_leaf_out": ["leaf_out"],
# }
#
# # Search the row fields for flowering fruiting leaf_out in this order
# SEARCH_FIELDS = """
#     reproductivecondition occurrenceremarks fieldnotes dynamicproperties """.split()
#
#
# def filter_records(in_db: Path, out_db: Path, filter_set: str) -> None:
#     """Remove records that are not angiosperms, have no phenological data, etc."""
#     renames = get_column_renames()
#     in_sql = build_select(renames)
#
#     nlp = pipeline.pipeline()
#
#     batch = []
#
#     with sqlite3.connect(in_db) as in_cxn, sqlite3.connect(out_db) as out_cxn:
#         in_cxn.row_factory = sqlite3.Row
#
#         all_columns = list(renames.values()) + ["filter_set"] + load.TRAITS
#         create_angiosperms_table(out_cxn, all_columns)
#
#         for raw in tqdm(in_cxn.execute(in_sql)):
#             row = dict(raw)
#             row["filter_set"] = filter_set
#
#             # Add empty traits to the record
#             for field in load.TRAITS:
#                 row[field] = OFF
#
#             for field in SEARCH_FIELDS:
#                 if set_row_flags(row, nlp, field):
#                     batch.append(row)
#                     break
#
#     db.insert_batch(out_db, build_insert(renames), batch)
#
#
# def set_row_flags(row, nlp, field):
#     """Set the row flags for traits."""
#     flags_set = False
#     if not row[field]:
#         return flags_set
#
#     doc = nlp(row[field])
#     for ent in doc.ents:
#         trait = ent._.data["trait"]
#
#         flags = REPRO_LABELS if field == "reproductivecondition" else LABELS
#
#         for flag in flags.get(trait, []):
#             row[flag] = ON
#             flags_set = True
#
#     return flags_set
#
#
# def get_column_renames():
#     """Get the output column names."""
#     columns = sorted(set(load.MULTIMEDIA + load.OCCURRENCE + load.OCCURRENCE_RAW))
#     renames = {c: c.split(":")[-1].lower() for c in columns}
#     renames["dwc:class"] = "class_"
#     renames["dwc:order"] = "order_"
#     return renames
#
#
# def build_select(renames):
#     """Build the select statement for reading iDigBio records."""
#     columns = [f"`{k}` as {v}" for k, v in renames.items()]
#     fields = ", ".join(columns)
#     sql = f"""
#         with multiples as (
#             select coreid
#               from multimedia
#           group by coreid
#             having count(*) > 1
#         )
#         select {fields}
#         from multimedia
#         join occurrence using (coreid)
#         join occurrence_raw using (coreid)
#        where coreid not in (select coreid from multiples)
#          and accessuri <> ''
#          and `dwc:basisofrecord` <> 'fossilspecimen'
#          and `dwc:order` in (select order_ from orders);
#        """
#     return sql
#
#
# def build_insert(renames):
#     """Build an insert statement for the angiosperm records."""
#     columns = list(renames.values()) + ["filter_set"] + load.TRAITS
#     fields = ", ".join(columns)
#     values = [f":{f}" for f in columns]
#     values = ", ".join(values)
#     sql = f""" insert into angiosperms ({fields}) values ({values}); """
#     return sql
#
#
# def create_angiosperms_table(out_cxn, columns):
#     """Create the angiosperm table."""
#     fields = [f"            {c} text" for c in columns if c != "coreid"]
#     fields = ",\n".join(fields)
#
#     sql = f"""
#         create table if not exists angiosperms (
#             coreid text primary key,
#             {fields}
#         );
#     """
#     out_cxn.executescript(sql)
#
#
# def parse_args() -> argparse.Namespace:
#     """Process command-line arguments."""
#     description = """TODO: Rewrite this to allow multiple NLP runs and records without
#         notations."""
#
#     arg_parser = argparse.ArgumentParser(
#         description=textwrap.dedent(description), fromfile_prefix_chars="@"
#     )
#
#     arg_parser.add_argument(
#         "--in-db",
#         metavar="PATH",
#         type=Path,
#         required=True,
#         help="""Path to the input SQLite3 database (iDigBio data).""",
#     )
#
#     arg_parser.add_argument(
#         "--out-db",
#         metavar="PATH",
#         type=Path,
#         required=True,
#         help="""Path to the output SQLite3 database (angiosperm data).""",
#     )
#
#     default = datetime.now().isoformat(sep="_", timespec="seconds")
#     arg_parser.add_argument(
#         "--filter-set",
#         default=default,
#         help="""Name the filter set to allow multiple extracts.""",
#     )
#
#     args = arg_parser.parse_args()
#     return args
#
#
# def main() -> None:
#     """Filter the records."""
#     log.started()
#     args = parse_args()
#     filter_records(args.in_db, args.out_db, args.filter_set)
#     log.finished()
#
#
# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python
# """Filter iDigBio records."""
# import argparse
# import sqlite3
# import textwrap
# from datetime import datetime
# from pathlib import Path
#
# from pylib import db
# from pylib import log
# from pylib import pipeline
# from tqdm import tqdm
#
# from junk import idigbio_load as load
#
# ON, OFF = 1, ""
#
# LABELS = {
#     "flowering": ["flowering"],
#     "flowering_fruiting": ["flowering", "fruiting"],
#     "fruiting": ["fruiting"],
#     "leaf_out": ["leaf_out"],
#     "not_flowering": ["not_flowering"],
#     "not_fruiting": ["not_fruiting"],
#     "not_leaf_out": ["not_leaf_out"],
# }
#
# REPRO_LABELS = LABELS | {
#     "abbrev_flowering": ["flowering"],
#     "abbrev_flowering_fruiting": ["flowering", "fruiting"],
#     "abbrev_fruiting": ["fruiting"],
#     "abbrev_leaf_out": ["leaf_out"],
# }
#
# # Search the row fields for flowering fruiting leaf_out in this order
# SEARCH_FIELDS = """
#     reproductivecondition occurrenceremarks fieldnotes dynamicproperties """.split()
#
#
# def filter_records(in_db: Path, out_db: Path, filter_set: str) -> None:
#     """Remove records that are not angiosperms, have no phenological data, etc."""
#     renames = get_column_renames()
#     in_sql = build_select(renames)
#
#     nlp = pipeline.pipeline()
#
#     batch = []
#
#     with sqlite3.connect(in_db) as in_cxn, sqlite3.connect(out_db) as out_cxn:
#         in_cxn.row_factory = sqlite3.Row
#
#         all_columns = list(renames.values()) + ["filter_set"] + load.TRAITS
#         create_angiosperms_table(out_cxn, all_columns)
#
#         for raw in tqdm(in_cxn.execute(in_sql)):
#             row = dict(raw)
#             row["filter_set"] = filter_set
#
#             # Add empty traits to the record
#             for field in load.TRAITS:
#                 row[field] = OFF
#
#             for field in SEARCH_FIELDS:
#                 if set_row_flags(row, nlp, field):
#                     batch.append(row)
#                     break
#
#     db.insert_batch(out_db, build_insert(renames), batch)
#
#
# def set_row_flags(row, nlp, field):
#     """Set the row flags for traits."""
#     flags_set = False
#     if not row[field]:
#         return flags_set
#
#     doc = nlp(row[field])
#     for ent in doc.ents:
#         trait = ent._.data["trait"]
#
#         flags = REPRO_LABELS if field == "reproductivecondition" else LABELS
#
#         for flag in flags.get(trait, []):
#             row[flag] = ON
#             flags_set = True
#
#     return flags_set
#
#
# def get_column_renames():
#     """Get the output column names."""
#     columns = sorted(set(load.MULTIMEDIA + load.OCCURRENCE + load.OCCURRENCE_RAW))
#     renames = {c: c.split(":")[-1].lower() for c in columns}
#     renames["dwc:class"] = "class_"
#     renames["dwc:order"] = "order_"
#     return renames
#
#
# def build_select(renames):
#     """Build the select statement for reading iDigBio records."""
#     columns = [f"`{k}` as {v}" for k, v in renames.items()]
#     fields = ", ".join(columns)
#     sql = f"""
#         with multiples as (
#             select coreid
#               from multimedia
#           group by coreid
#             having count(*) > 1
#         )
#         select {fields}
#         from multimedia
#         join occurrence using (coreid)
#         join occurrence_raw using (coreid)
#        where coreid not in (select coreid from multiples)
#          and accessuri <> ''
#          and `dwc:basisofrecord` <> 'fossilspecimen'
#          and `dwc:order` in (select order_ from orders);
#        """
#     return sql
#
#
# def build_insert(renames):
#     """Build an insert statement for the angiosperm records."""
#     columns = list(renames.values()) + ["filter_set"] + load.TRAITS
#     fields = ", ".join(columns)
#     values = [f":{f}" for f in columns]
#     values = ", ".join(values)
#     sql = f""" insert into angiosperms ({fields}) values ({values}); """
#     return sql
#
#
# def create_angiosperms_table(out_cxn, columns):
#     """Create the angiosperm table."""
#     fields = [f"            {c} text" for c in columns if c != "coreid"]
#     fields = ",\n".join(fields)
#
#     sql = f"""
#         create table if not exists angiosperms (
#             coreid text primary key,
#             {fields}
#         );
#     """
#     out_cxn.executescript(sql)
#
#
# def parse_args() -> argparse.Namespace:
#     """Process command-line arguments."""
#     description = """TODO: Rewrite this to allow multiple NLP runs and records without
#         notations."""
#
#     arg_parser = argparse.ArgumentParser(
#         description=textwrap.dedent(description), fromfile_prefix_chars="@"
#     )
#
#     arg_parser.add_argument(
#         "--in-db",
#         metavar="PATH",
#         type=Path,
#         required=True,
#         help="""Path to the input SQLite3 database (iDigBio data).""",
#     )
#
#     arg_parser.add_argument(
#         "--out-db",
#         metavar="PATH",
#         type=Path,
#         required=True,
#         help="""Path to the output SQLite3 database (angiosperm data).""",
#     )
#
#     default = datetime.now().isoformat(sep="_", timespec="seconds")
#     arg_parser.add_argument(
#         "--filter-set",
#         default=default,
#         help="""Name the filter set to allow multiple extracts.""",
#     )
#
#     args = arg_parser.parse_args()
#     return args
#
#
# def main() -> None:
#     """Filter the records."""
#     log.started()
#     args = parse_args()
#     filter_records(args.in_db, args.out_db, args.filter_set)
#     log.finished()
#
#
# if __name__ == "__main__":
#     main()
