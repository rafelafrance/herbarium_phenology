"""Common function for dealing with model training, validation, or inference."""
import sys

from . import db


def validate_split_set(args):
    """Make sure that the entered split set is in the database."""
    split_sets = db.select_all_split_sets(args.database)
    if args.split_set not in split_sets:
        print(f"'{args.split_set}' is not in split_runs. Valid split_sets:")
        print(", ".join(split_sets))
        sys.exit(1)


def validate_target_set(args):
    """Make sure that the entered split set is in the database."""
    target_sets = db.select_all_target_sets(args.database)
    if args.target_set not in target_sets:
        print(f"'{args.target_set}' is not in targets. Valid target_sets:")
        print(", ".join(target_sets))
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
