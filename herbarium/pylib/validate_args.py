"""Validate input arguments."""
import sys

from . import db
from . import db_new


def validate_inference_set(database, inference_set):
    """Make sure that the entered split set is in the database."""
    inference_sets = [
        r["inference_set"] for r in db_new.canned_select(database, "all_inference_sets")
    ]
    if inference_set not in inference_sets:
        err = f"'{inference_set}' is not in inference sets. Valid inference_sets:\n"
        err += ", ".join(inference_sets)
        sys.exit(err)


def validate_split_set(database, split_set):
    """Make sure that the entered split set is in the database."""
    split_sets = [
        r["split_set"] for r in db_new.canned_select(database, "all_split_sets")
    ]
    if split_set not in split_sets:
        err = f"'{split_set}' is not in split sets. Valid split_sets:\n"
        err += ", ".join(split_sets)
        sys.exit(err)


def validate_target_set(database, target_set):
    """Make sure that the entered split set is in the database."""
    target_sets = [
        r["target_set"] for r in db_new.canned_select(database, "all_target_sets")
    ]
    if target_set not in target_sets:
        err = f"'{target_set}' is not in targets. Valid target_sets:\n"
        err += ", ".join(target_sets)
        sys.exit(err)


def validate_orders(args):
    """Make sure the entered orders are in the database."""
    if not args.order:
        return

    orders = db.select_all_orders(args.database)
    bad = [o for o in args.order if o not in orders]
    if bad:
        err = "Orders not in the database:\n"
        err += ", ".join(bad)
        err += "\nValid orders:\n"
        err += ", ".join(orders)
        sys.exit(err)
