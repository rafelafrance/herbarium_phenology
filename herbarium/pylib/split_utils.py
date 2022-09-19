"""Split labeled images into training, testing, and validation datasets."""
from tqdm import tqdm

from . import db


def assign_records(args, orders):
    """Assign records to splits.

    We want to distribute the database records over the orders and targets in proportion
    to the desired distribution as much as possible.
    """
    db.canned_delete(args.database, "splits", split_set=args.split_set)

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
                where target_set = :target_set
                  and trait = :trait
                  and order_ = :order
                  and target = :target
             order by random()
            """
            rows = db.select(
                args.database,
                sql,
                target_set=args.target_set,
                trait=args.trait,
                order_=order,
                target=target,
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

            db.canned_insert(args.database, "splits", batch)


def extend_split_set(database, split_set, base_split_set, target_set, trait):
    """Start with this as the base split set."""
    if not base_split_set:
        return set()

    sql = """
        select split_set, split, coreid
          from splits
          join images using (coreid)
          join targets using (coreid)
         where split_set = :split_set
           and target_set = :target_set
           and trait = :trait
    """
    batch = db.select(
        database,
        sql,
        split_set=base_split_set,
        target_set=target_set,
        trait=trait,
    )
    for row in batch:
        row["split_set"] = split_set

    db.canned_insert(database, "splits", batch)

    return {r["coreid"] for r in batch}
