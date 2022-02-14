"""Split labeled images into training, testing, and validation datasets."""
import sqlite3

from tqdm import tqdm

from herbarium.pylib import db


def assign_records(args, orders):
    """Assign records to splits.

    We want to distribute the database records over the orders and targets in proportion
    to the desired distribution as much as possible.
    """
    delete_split_set(args.database, args.split_set)

    used = extend_split_set(
        args.database, args.split_set, args.base_split_set, args.target_set, args.trait
    )

    for order in tqdm(orders):
        for target in (0.0, 1.0):
            sql = """
               select coreid
                 from targets
                 join images using (coreid)
                 join angiosperms using (coreid)
                where target_set = ?
                  and trait = ?
                  and order_ = ?
                  and target = ?
             order by random()
            """
            rows = db.rows_as_dicts(
                args.database, sql, [args.target_set, args.trait, order, target]
            )

            coreids = {row["coreid"] for row in rows} - used
            used |= coreids

            batch = [{"split_set": args.split_set, "coreid": i} for i in coreids]

            count = len(coreids)

            # Try to make sure we get a validation record
            val_split = round(count * (args.test_split + args.val_split))
            if val_split <= 2 and count >= 2:
                val_split = 2

            # Try to make sure we get a test record
            test_split = round(count * args.test_split)
            if test_split == 0 and count >= 1:
                test_split = 1

            # Distribute the records
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
    if not base_split_set:
        return set()

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
