#!/usr/bin/python3

import csv
import sqlite3
from argparse import ArgumentParser


def upperFirst(strval):
    if len(strval) < 1:
        return strval
    else:
        return strval[0].upper() + strval[1:]


argp = ArgumentParser()
argp.add_argument(
    'sqlite_db', type=str, help='A SQLite database file.'
)
argp.add_argument(
    'output_csv_base', type=str,
    help='The base name to use for the output CSV files.'
)

args = argp.parse_args()

# This query gets the total number of records for each species.
query_1 = """SELECT
    ("dwc:genus" || ' ' || "dwc:specificEpithet") as species, count(*) AS records
    FROM occurrence
    GROUP BY "dwc:genus", "dwc:specificEpithet"
    ORDER BY records desc;"""

# This query gets the total number of records for each genus.
query_2 = """SELECT
    "dwc:genus" as genus, count(*) AS records
    FROM occurrence
    GROUP BY "dwc:genus"
    ORDER BY records desc;"""

# This query gets the total number of records for each family.
query_3 = """SELECT
    "dwc:family" as family, count(*) AS records
    FROM occurrence
    GROUP BY "dwc:family"
    ORDER BY records desc;"""

# This query gets the total number of records for each order.
query_4 = """SELECT
    "dwc:order" as "order", count(*) AS records
    FROM occurrence
    GROUP BY "dwc:order"
    ORDER BY records desc;"""

query_sets = [
    ('species', query_1),
    ('genus', query_2),
    ('family', query_3),
    ('order', query_4)
]

conn = sqlite3.connect(args.sqlite_db)
conn.row_factory = sqlite3.Row
c = conn.cursor()

for qset in query_sets:
    rowout = {}
    fout_name = args.output_csv_base + qset[0] + '.csv'

    with open(fout_name, 'w') as fout:
        writer = csv.DictWriter(fout, [qset[0], 'records'])
        writer.writeheader()

        print('Analyzing {0} counts...'.format(qset[0]))
        c.execute(qset[1])
        for r in c:
            rowout[qset[0]] = upperFirst(r[qset[0]])
            rowout['records'] = r['records']
            writer.writerow(rowout)

c.close()

