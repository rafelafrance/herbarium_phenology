#!/usr/bin/python3

import csv
import sys
from argparse import ArgumentParser


csv.field_size_limit(sys.maxsize)

argp = ArgumentParser(
    description='Generates a report of missing family and genus names in one '
    'or more occurrence.csv data files from iDigBio.'
)
argp.add_argument(
    'input_CSV', type=str, nargs='+',
    help='An input iDigBio occurrence.csv.'
)

args = argp.parse_args()

missing_family = 0
missing_genus = 0
missing_both = 0
rowcnt = 0

for fname in args.input_CSV:
    with open(fname) as fin:
        reader = csv.DictReader(fin)
        try:
            for row in reader:
                if row['dwc:family'].strip() == '':
                    missing_family += 1

                if row['dwc:genus'].strip() == '':
                    missing_genus += 1

                if (
                    row['dwc:family'].strip() == '' and
                    row['dwc:genus'].strip() == ''
                ):
                    missing_both += 1

                rowcnt += 1
                if rowcnt % 100000 == 0:
                    print('{0:,} rows processed...'.format(rowcnt), file=sys.stderr)
        except Exception as err:
            print(f'Error encountered at row {rowcnt:,}: {err}.', file=sys.stderr)

print(f'Total records: {rowcnt:,}')
print('Records with no family information: {0:,} ({1}%)'.format(
    missing_family, round((missing_family / rowcnt) * 100, 2)
))
print('Records with no genus information: {0:,} ({1}%)'.format(
    missing_genus, round((missing_genus / rowcnt) * 100, 2)
))
print('Records with no genus or family information: {0:,} ({1}%)'.format(
    missing_both, round((missing_both / rowcnt) * 100, 2)
))

