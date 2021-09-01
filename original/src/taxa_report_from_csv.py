#!/usr/bin/python3

import csv
import sys
from argparse import ArgumentParser


csv.field_size_limit(sys.maxsize)

argp = ArgumentParser(
    description='Generates a report of the frequency of all class, order, and '
    'family names in one or more occurrence.csv data files from iDigBio.'
)
argp.add_argument(
    'input_CSV', type=str, nargs='+',
    help='An input iDigBio occurrence.csv.'
)

args = argp.parse_args()

classes = {}
orders = {}
families = {}
rowcnt = 0

for fname in args.input_CSV:
    with open(fname) as fin:
        reader = csv.DictReader(fin)
        try:
            for row in reader:
                classname = row['dwc:class']
                if classname not in classes:
                    classes[classname] = 0
                classes[classname] += 1

                ordername = row['dwc:order']
                if ordername not in orders:
                    orders[ordername] = 0
                orders[ordername] += 1

                familyname = row['dwc:family']
                if familyname not in families:
                    families[familyname] = 0
                families[familyname] += 1

                rowcnt += 1
                if rowcnt % 100000 == 0:
                    print('{0:,} rows processed...'.format(rowcnt), file=sys.stderr)
        except Exception as err:
            print(f'Error encountered at row {rowcnt:,}: {err}.', file=sys.stderr)

print('**classes**')
for classname in sorted(list(classes.keys())):
    print('{0}: {1}'.format(classname, classes[classname]))

print('\n**orders**')
for ordername in sorted(list(orders.keys())):
    print('{0}: {1}'.format(ordername, orders[ordername]))

print('\n**families**')
for familyname in sorted(list(families.keys())):
    print('{0}: {1}'.format(familyname, families[familyname]))

