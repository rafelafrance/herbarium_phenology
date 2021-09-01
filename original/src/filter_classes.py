#!/usr/bin/python3

import csv
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Filters out non-angiosperm classes from a CSV file of '
    'iDigBio records.'
)
argp.add_argument(
    'input_CSV', type=str, help='An input iDigBio occurrence.csv.'
)
argp.add_argument(
    'output_CSV', type=str, help='An output CSV file.'
)

args = argp.parse_args()

exclude_classes = set((
    'bennititopsida', 'cycadopsida', 'equisetopsida', 'filicopsida',
    'ginkgoopsida', 'gnetopsida', 'gymnospermopsida', 'isoetopsida',
    'lepidophytopsida', 'lycopodiopsida', 'lycopsida', 'marattiopsida',
    'pinopsida', 'polypodiopsida', 'progymnospermopsida', 'psilotopsida',
    'pteridospermopsida', 'rhyniopsida', 'sphenopsida', 'trimerophytopsida',
    'zosterophyllopsida'
))
include_classes = set((
    'angiospermopsida', 'dicotyledonae', 'liliopsida', 'magnoliopsida',
    'monocots', 'monocotyledonae'
))

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

        if row['dwc:class'].lower() in include_classes:
            keptcnt += 1
            writer.writerow(row)

        if rowcnt % 100000 == 0:
            print('{0:,} rows processed...'.format(rowcnt))

print('\nProcessed {0:,} rows, kept {1:,} rows.'.format(rowcnt, keptcnt))

