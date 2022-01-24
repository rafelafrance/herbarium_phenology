"""Common function for dealing with model training, validation, or inference."""
import sys

from . import db


def validate_split_runs(args):
    """Make sure that the entered split set is in the database."""
    split_runs = db.select_all_split_sets(args.database)
    if args.split_set not in split_runs:
        print(f"'{args.split_set}' is not in split_runs. Valid split_runs:")
        print(", ".join(split_runs))
        sys.exit(1)


def validate_orders(args):
    """Make sure the entered orders are in the database."""
    if not args.order:
        return

    orders = db.select_all_orders(args.database)
    bad = [o for o in args.order if o not in orders]
    if bad:
        print("Orders not in the database:")
        print(", ".join(bad))
        print("Valid orders:")
        print(", ".join(orders))
        sys.exit(1)
