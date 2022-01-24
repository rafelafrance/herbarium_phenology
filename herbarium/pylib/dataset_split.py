"""Utilities for splitting images datasets into training, validation, & testing sets."""
from tqdm import tqdm

from . import const
from . import db


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
