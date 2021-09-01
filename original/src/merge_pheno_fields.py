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

#
# @author: Brian Stucky
#

import csv
import sqlite3
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Searches for possible phenological information in raw '
    'iDigBio data and otpionally adds relevant data fields to an occurrences '
    'database.'
)
argp.add_argument(
    '-d', '--sqlite_db', type=str, required=False, default=None,
    help='A SQLite database to which to add phenological information.'
)
argp.add_argument(
    'input_CSV', type=str, nargs='+',
    help='An input iDigBio occurrences_raw.csv.'
)

args = argp.parse_args()


# Search terms for identifying phenological information.
ph_terms = (
    'flower', 'fruit', 'petal', 'fls.', 'corolla', 'leaves', 'tepal', 'seed',
    'sterile', 'ray', 'infl.', 'bract', 'inflor.', 'inflorescence', 'stigma',
    'sepal', 'flores'
)


def containsPhenoInfo(text_str):
    text_str = text_str.lower()

    for ph_term in ph_terms:
        if ph_term in text_str:
            return True

    return False


if args.sqlite_db is not None:
    conn = sqlite3.connect(args.sqlite_db)
    c = conn.cursor()
else:
    c = None

rowcnts = {
    'all': 0,
    'valid': 0,
    'has_relevant': 0,
    'has_possible': 0,
    'has_repro_condition': 0
}

update_sql = """UPDATE "occurrence"
SET
    "dwc:reproductiveCondition"=?,
    "dwc:occurrenceRemarks"=?,
    "dwc:dynamicProperties"=?,
    "dwc:fieldNotes"=?
WHERE
    "coreid"=?
"""

for input_CSV in args.input_CSV:
    print('Processing {0}...'.format(input_CSV))

    with open(input_CSV) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            coreid = row['coreid']
            rowcnts['all'] += 1

            if row['dwc:genus'] != '':
                rowcnts['valid'] += 1

            if (
                row['dwc:reproductiveCondition'] != '' or
                row['dwc:occurrenceRemarks'] != '' or
                row['dwc:dynamicProperties'] != '' or
                row['dwc:fieldNotes'] != ''
            ):
                rowcnts['has_relevant'] += 1
                if row['dwc:reproductiveCondition'] != '':
                    rowcnts['has_repro_condition'] += 1

                if (
                    row['dwc:reproductiveCondition'] != '' or
                    containsPhenoInfo(row['dwc:occurrenceRemarks']) or
                    containsPhenoInfo(row['dwc:dynamicProperties']) or
                    containsPhenoInfo(row['dwc:fieldNotes'])
                ):
                    rowcnts['has_possible'] += 1

                if c is not None:
                    c.execute(
                        update_sql, [
                            row['dwc:reproductiveCondition'],
                            row['dwc:occurrenceRemarks'],
                            row['dwc:dynamicProperties'],
                            row['dwc:fieldNotes'],
                            coreid
                        ]
                    )

            if rowcnts['all'] % 100000 == 0:
                print('  {0:,} rows processed...'.format(
                    rowcnts['all']
                ))
                if c is not None:
                    conn.commit()

print('{0:,} rows processed.'.format(rowcnts['all']))
print('{0:,} valid rows.'.format(rowcnts['valid']))
print('{0:,} rows ({1}%) with one or more relevant data fields.'.format(
    rowcnts['has_relevant'],
    round((rowcnts['has_relevant'] / rowcnts['valid']) * 100, 2)
))
print('{0:,} rows ({1}%) with potential phenological information.'.format(
    rowcnts['has_possible'],
    round((rowcnts['has_possible'] / rowcnts['valid']) * 100, 2)
))
print('{0:,} rows ({1}%) with "dwc:reproductiveCondition".'.format(
    rowcnts['has_repro_condition'],
    round((rowcnts['has_repro_condition'] / rowcnts['valid']) * 100, 2)
))

if c is not None:
    conn.commit()
    conn.close()

