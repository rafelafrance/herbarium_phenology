#!/usr/bin/env python3
"""Create split runs from images."""
import argparse
import sqlite3
import textwrap
from pathlib import Path

from pylib import db
from pylib import validate_args as val
from pylib.const import ALL_TRAITS
from tqdm import tqdm


def assign_records(args, orders):
    """Assign records to splits.

    We want to distribute the database records over the orders and traits in proportion
    to the actual distribution as possible.
    """
    delete_split_set(args.database, args.split_set)

    used = extend_split_set(
        args.database, args.split_set, args.base_split_set, args.target_set, args.trait
    )

    for order in tqdm(orders):
        sql = """
           select coreid
             from targets
             join images using (coreid)
             join angiosperms using (coreid)
            where target_set = ?
              and trait = ?
              and order_ = ?
         order by random()
        """
        rows = db.rows_as_dicts(
            args.database, sql, [args.target_set, args.trait, order]
        )

        coreids = {row["coreid"] for row in rows} - used
        used |= coreids

        batch = [{"split_set": args.split_set, "coreid": i} for i in coreids]

        count = len(coreids)

        test_split = round(count * args.test_split)
        val_split = round(count * (args.test_split + args.val_split))

        for i in range(count):
            if i <= test_split:
                split = "test"
            elif i <= val_split:
                split = "val"
            else:
                split = "train"

            batch[i]["split"] = split

        db.insert_splits(args.database, batch)


def extend_split_set(database, split_set, base_split_set, target_set, trait):
    """Start with this as the base split set."""
    sql = """
        select split_set, split, coreid
          from splits
          join images using (coreid)
          join targets using (coreid)
         where split_set = ?
           and target_set = ?
           and trait = ?
    """
    batch = db.rows_as_dicts(database, sql, [base_split_set, target_set, trait])
    for row in batch:
        row["split_set"] = split_set

    db.insert_splits(database, batch)

    return {r["coreid"] for r in batch}


def delete_split_set(database, split_set):
    """Remove the old split set before adding new data."""
    sql = """delete from splits where split_set = ?"""
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, (split_set,))


def parse_args():
    """Process command-line arguments."""
    description = """Split images into training, validation, and test sets."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--database",
        "--db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the SQLite3 database (angiosperm data).""",
    )

    arg_parser.add_argument(
        "--split-set",
        metavar="NAME",
        required=True,
        help="""Give the split set this name.""",
    )

    arg_parser.add_argument(
        "--target-set",
        metavar="NAME",
        required=True,
        help="""Use this target set for target values.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=ALL_TRAITS,
        required=True,
        help="""Which trait to classify.""",
    )

    arg_parser.add_argument(
        "--base-split-set",
        metavar="NAME",
        help="""Start with this split set. Add split records from this set before
            adding any new records from the --target-set. The is so we don't train
            on a previous test data when updating a model.""",
    )

    arg_parser.add_argument(
        "--train-split",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the validation. I.e. evaluating
            training progress at the end of each epoch. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the testing. I.e. the holdout
            data used to evaluate the model after training. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()

    val.validate_target_set(args.database, args.target_set)
    if args.base_split_set:
        val.validate_split_set(args.database, args.base_split_set)

    return args


def main():
    """Infer traits."""
    args = parse_args()
    orders = db.select_all_orders(args.database)
    assign_records(args, orders)


if __name__ == "__main__":
    main()
