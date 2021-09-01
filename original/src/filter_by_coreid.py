#!/usr/bin/python3

# Filters out iDigBio records by comparing to a reference set of coreids.

import csv
import sys


if len(sys.argv) != 4:
    exit('\nERROR: Usage: filter_classes.py CSV_COREID_REF CSV_IN CSV_OUT\n')

# Build the set of coreids.
coreids = set()
with open(sys.argv[1]) as fin:
    reader = csv.DictReader(fin)
    print('Reading reference core IDs...')
    for cnt, row in enumerate(reader):
        coreids.add(row['coreid'])
        if (cnt + 1) % 100000 == 0:
            print('  {0:,} rows processed...'.format(cnt + 1))
print('done.')

# Get the column names.
with open(sys.argv[2]) as fin:
    reader = csv.reader(fin)
    colnames = reader.__next__()

rowcnt = keptcnt = 0

with open(sys.argv[2]) as fin, open(sys.argv[3], 'w') as fout:
    print('Filtering input CSV rows...')
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, colnames)
    writer.writeheader()

    for row in reader:
        rowcnt += 1
        if row['coreid'] in coreids:
            keptcnt += 1
            writer.writerow(row)
        if rowcnt % 100000 == 0:
            print('  {0:,} rows processed...'.format(rowcnt))
print('done.')

print('Processed {0:,} rows, kept {1:,} rows.'.format(rowcnt, keptcnt))

