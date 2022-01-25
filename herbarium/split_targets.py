#!/usr/bin/env python3
"""Create split runs from images."""
import argparse
import textwrap
from pathlib import Path

from pylib import const
from pylib import db
from pylib.const import ALL_TRAITS
from tqdm import tqdm


def assign_records(args):
    """Assign records to splits.

    We want to distribute the database records over the orders and traits in proportion
    to the actual distribution as possible.
    """
    orders = db.select_all_orders(args.database)
    used = set()

    for order in tqdm(orders):
        for trait in const.ALL_TRAIT_FIELDS:
            sql = f"""
               select coreid
                 from images join angiosperms using (coreid)
                where order_ = ? and {trait} = 1
             order by random() """
            rows = db.rows_as_dicts(args.database, sql, [order])

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
    return args


def main():
    """Infer traits."""
    args = parse_args()
    if not args.trait:
        args.trait = ALL_TRAITS

    assign_records(args)


if __name__ == "__main__":
    main()
