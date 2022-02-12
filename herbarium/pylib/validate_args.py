"""Common function for dealing with model training, validation, or inference."""
import sys

from . import db


def validate_split_set(database, split_set):
    """Make sure that the entered split set is in the database."""
    split_sets = db.select_all_split_sets(database)
    if split_set not in split_sets:
        print(f"'{split_set}' is not in split sets. Valid split_sets:")
        print(", ".join(split_sets))
        sys.exit(1)


def validate_target_set(database, target_set):
    """Make sure that the entered split set is in the database."""
    target_sets = db.select_all_target_sets(database)
    if target_set not in target_sets:
        print(f"'{target_set}' is not in targets. Valid target_sets:")
        print(", ".join(target_sets))
        sys.exit(1)


def validate_inference_set(database, inference_set):
    """Make sure that the entered split set is in the database."""
    inference_sets = db.select_all_inference_sets(database)
    if inference_set not in inference_sets:
        print(f"'{inference_set}' is not in inference sets. Valid inference_sets:")
        print(", ".join(inference_sets))
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
