#!/usr/bin/python3

# Copyright (C) 2021 Brian J. Stucky
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Converts iDigBio results to a SQLite database.
#
# @author: Brian Stucky
#

import csv
import sqlite3
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Converts iDigBio results to a SQLite database (which need not '
    'exist yet).  Automatically filters out fossil specimen records and '
    'records without a species-level identification.  For coreids with more '
    'than one media object, only the first-encountered media object will be '
    'added to the database.'
)
argp.add_argument(
    '-o', '--occurrence', type=str, action='append', required=True,
    help='The path to an input CSV file with the structure of "occurrence.csv" '
    'from an iDigBio data download archive.'
)
argp.add_argument(
    '-m', '--multimedia', type=str, action='append', required=True,
    help='The path to an input CSV file with the structure of "multimedia.csv" '
    'from an iDigBio data download archive.'
)
argp.add_argument(
    'sqlite_db', type=str, help='A SQLite database file.'
)

args = argp.parse_args()

if len(args.occurrence) != len(args.multimedia):
    exit('ERROR: Unequal numbers of occurrence and multimedia input files.')

cols_to_keep = (
    'coreid', 'dwc:basisOfRecord', 'dwc:order', 'dwc:family', 'dwc:genus',
    'dwc:specificEpithet', 'dwc:scientificName', 'dwc:eventDate',
    'dwc:continent', 'dwc:country', 'dwc:stateProvince', 'dwc:county',
    'dwc:locality', 'idigbio:geoPoint'
)
extra_cols = (
    'dwc:reproductiveCondition', 'dwc:occurrenceRemarks',
    'dwc:dynamicProperties', 'dwc:fieldNotes', 'hasPhenoInfo', 'imgURI'
)
table_name = 'occurrence'

conn = sqlite3.connect(args.sqlite_db)
c = conn.cursor()
ct_sql = 'CREATE TABLE {0} ({1})'.format(
    table_name,
    ','.join(['"{0}" TEXT'.format(colname) for colname in cols_to_keep + extra_cols])
)
# Use integers for hasPhenoInfo.
ct_sql = ct_sql.replace(
    '"hasPhenoInfo" TEXT', '"hasPhenoInfo" INTEGER DEFAULT 0'
)
# Use coreid as the primary key.  Note that we use "REPLACE" to resolve coreid
# key conflicts.  This is because I found that such conflicts do occur in
# iDigBio result sets that are supposed to be disjoint (!), so we can only
# accept one record per coreid.
ct_sql = ct_sql.replace(
    '"coreid" TEXT', '"coreid" TEXT PRIMARY KEY ON CONFLICT REPLACE'
)

c.execute(ct_sql)

insert_tmpl = 'INSERT INTO {0} ({1}) VALUES ({2})'.format(
    table_name,
    ','.join(['"{0}"'.format(colname) for colname in cols_to_keep]),
    ','.join(['?'] * len(cols_to_keep))
)

rowcnts = {}
coreids = set()

for occ_csv in args.occurrence:
    print('Processing {0}...'.format(occ_csv))
    rowcnts[occ_csv] = 0

    with open(occ_csv) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if (
                row['dwc:basisOfRecord'] != 'fossilspecimen' and
                row['dwc:genus'] != ''# and
                #row['dwc:eventDate'] != '' and
                #row['idigbio:geoPoint'] != ''
            ):
                rowcnts[occ_csv] += 1

                row_vals = []
                for colname in cols_to_keep:
                    row_vals.append(row[colname])

                coreids.add(row['coreid'])

                c.execute(insert_tmpl, row_vals)

                if rowcnts[occ_csv] % 100000 == 0:
                    print('  {0:,} rows processed...'.format(
                        rowcnts[occ_csv]
                    ))
                    conn.commit()

for occ_csv in args.occurrence:
    print('{0}: {1:,} total rows processed.'.format(
        occ_csv, rowcnts[occ_csv]
    ))
print()
conn.commit()

# Add image URIs to the database.
update_tmpl = """UPDATE "occurrence"
SET
    "imgURI"=?
WHERE
    "coreid"=?
"""

rowcnts = {}

for mm_csv in args.multimedia:
    print('Processing {0}...'.format(mm_csv))
    rowcnts[mm_csv] = 0

    with open(mm_csv) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if row['coreid'] in coreids:
                rowcnts[mm_csv] += 1

                row_vals = [row['ac:accessURI'], row['coreid']]
                c.execute(update_tmpl, row_vals)

                coreids.remove(row['coreid'])

                if rowcnts[mm_csv] % 100000 == 0:
                    print('  {0:,} rows updated...'.format(
                        rowcnts[mm_csv]
                    ))
                    conn.commit()

for mm_csv in args.multimedia:
    print('{0}: {1:,} total rows updated.'.format(
        mm_csv, rowcnts[mm_csv]
    ))
print()
conn.commit()
conn.close()

if len(coreids) > 0:
    print('WARNING: Image URIs were not found for {0:,} coreids.\n'.format(
        len(coreids))
    )

