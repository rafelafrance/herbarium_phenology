#!/usr/bin/python3

import csv
import sys
import os.path
from argparse import ArgumentParser


def processNamesFile(filepath, nameset):
    """
    filepath: A path to a file containing family-level names.
    nameset: A set to which to add the names.
    """
    if os.path.splitext(filepath)[1] == '.txt':
        with open(filepath) as fin:
            for line in fin:
                nameset.add(line.strip().lower())
    elif os.path.splitext(filepath)[1] == '.csv':
        with open(filepath) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                nameset.add(row['name'].lower())
    else:
        raise Exception('Could not determine type of names file.')


csv.field_size_limit(sys.maxsize)

argp = ArgumentParser(
    description='Filters out non-angiosperm classes from a CSV file of '
    'iDigBio records.'
)
argp.add_argument(
    '-n', '--names_file', type=str, action='append', required=True,
    help='An input file of target family names. If the iinput file name '
    'matches *.txt, it will be interpreted as each line contains a single '
    'name. If the the name matches *.csv, it will be interpreted as a CSV '
    'file with a "name" column.'
)
argp.add_argument(
    'input_CSV', type=str, help='An input iDigBio occurrence.csv.'
)
argp.add_argument(
    'output_CSV', type=str, help='An output CSV file.'
)

args = argp.parse_args()

# Get the family-level names to use for record filtering.
family_names = set()
for names_file in args.names_file:
    processNamesFile(names_file, family_names)
#print(family_names)

# Get the column names.
with open(args.input_CSV) as fin:
    reader = csv.reader(fin)
    colnames = reader.__next__()

rowcnt = keptcnt = 0

with open(args.input_CSV) as fin, open(args.output_CSV, 'w') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, colnames)
    writer.writeheader()

    for row in reader:
        rowcnt += 1

        if row['dwc:family'].lower() in family_names:
            keptcnt += 1
            writer.writerow(row)

        if rowcnt % 100000 == 0:
            print('{0:,} rows processed...'.format(rowcnt))

print(
    '\nProcessed {0:,} rows, kept {1:,} rows ({2}%), discarded {3:,} rows '
    '({4}%).'.format(
        rowcnt, keptcnt, round((keptcnt / rowcnt) * 100, 2),
        rowcnt - keptcnt, round(((rowcnt - keptcnt) / rowcnt) * 100, 2)
    )
)

