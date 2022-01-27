"""Utilities for angiosperm.sqlite databases."""
import re
import sqlite3
import sys
from pathlib import Path
from typing import Union

DbPath = Union[Path, str]


def build_select(sql: str, *, limit: int = 0, **kwargs) -> tuple[str, list]:
    """Select records given a base SQL statement and keyword parameters."""
    sql, params = build_where(sql, **kwargs)

    if limit:
        sql += " limit ?"
        params.append(limit)

    return sql, params


def build_where(sql: str, **kwargs) -> tuple[str, list]:
    """Build a simple-mined where clause."""
    params, where = [], []

    for key, value in kwargs.items():
        key = key.strip("_")
        if value is None:
            pass
        elif isinstance(value, list) and value:
            where.append(f"{key} in ({','.join(['?'] * len(value))})")
            params += value
        else:
            where.append(f"{key} = ?")
            params.append(value)

    sql += (" where " + " and ".join(where)) if where else ""
    return sql, params


def rows_as_dicts(database: DbPath, sql: str, params: list = None) -> list[dict]:
    """Convert the SQL execute cursor to a list of dicts."""
    params = params if params else []
    with sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES) as cxn:
        cxn.row_factory = sqlite3.Row
        rows = [dict(r) for r in cxn.execute(sql, params)]
    return rows


def insert_batch(database: DbPath, sql: str, batch: list) -> None:
    """Insert a batch of records."""
    if batch:
        with sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES) as cxn:
            cxn.executemany(sql, batch)


def create_table(database: DbPath, sql: str, *, drop: bool = False) -> None:
    """Create a table."""
    flags = re.IGNORECASE | re.VERBOSE
    match = re.search(r" if \s+ not \s+ exists \s+ (\w+) ", sql, flags=flags)

    if not match:
        sys.exit(f"Could not parse create table for '{sql}'")

    table = match.group(1)

    with sqlite3.connect(database) as cxn:
        if drop:
            cxn.executescript(f"""drop table if exists {table};""")

        cxn.executescript(sql)


# ########### orders table #######################################################


def select_all_orders(database: DbPath) -> list[str]:
    """Get all orders with images."""
    sql = """select order_ from orders order by order_"""
    with sqlite3.connect(database) as cxn:
        orders = [r[0] for r in cxn.execute(sql)]
    return orders


# ########### angiosperms table  ######################################################


def create_angiosperms_index(database):
    """Make sure there is only one record per coreid."""
    sql = """create unique index angiosperms_coreid on angiosperms (coreid);"""
    with sqlite3.connect(database) as cxn:
        cxn.executescript(sql)


# ########### target traits table  ####################################################


def create_targets_table(database: DbPath, drop: bool = False) -> None:
    """Create a table to hold the results of NLP traits data."""
    sql = """
        create table if not exists targets (
            coreid     text,
            target_set text,
            source_set text,
            trait      text,
            target     real
        );

        create unique index targets_idx on targets (coreid, target_set, trait);
        """
    create_table(database, sql, drop=drop)


def insert_targets(database: DbPath, batch: list, target_set: str, trait: str) -> None:
    """Insert a batch of target records."""
    sql = "delete from targets where target_set = ? and trait = ?"
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, (target_set, trait))

    sql = """insert into targets
                    ( coreid,  target_set,  source_set,  trait,  target)
             values (:coreid, :target_set, :source_set, :trait, :target);"""
    insert_batch(database, sql, batch)


def select_all_target_sets(database: DbPath) -> list[str]:
    """Get all split runs in the database."""
    sql = """select distinct target_set from targets order by target_set"""
    with sqlite3.connect(database) as cxn:
        runs = [r[0] for r in cxn.execute(sql)]
    return runs


# ########### Image table ##########################################################


def create_images_table(database: DbPath, drop: bool = False) -> None:
    """Create a table with paths to the valid herbarium sheet images."""
    sql = """
        create table if not exists images (
            coreid text primary key,
            path   text unique,
            width  integer,
            height integer
        );
        """
    create_table(database, sql, drop=drop)


def insert_images(database: DbPath, batch: list) -> None:
    """Insert a batch of sheets records."""
    sql = """insert into images ( coreid,  path,  width,  height)
                         values (:coreid, :path, :width, :height);"""
    insert_batch(database, sql, batch)


def select_images(database: DbPath, limit: int = 0) -> list[dict]:
    """Select all images."""
    sql = """select *
               from images
               join angiosperms using (coreid)
              where order_ in (select order_ from orders)
            """
    sql, params = build_select(sql, limit=limit)
    return rows_as_dicts(database, sql, params)


# ########### Split table ##########################################################


def create_splits_table(database: DbPath, drop: bool = False) -> None:
    """Create train/validation/test splits of the data.

    This is so I don't wind up training on my test data. Because an image can belong
    to multiple classes I need to be careful that I don't add any core IDs in the
    test split to the training/validation splits.
    """
    sql = """
        create table if not exists splits (
            split_set text,
            split     text,
            coreid    text
        );
        create unique index splits_idx on splits (split_set, coreid);
        """
    create_table(database, sql, drop=drop)


def insert_splits(database: DbPath, batch: list) -> None:
    """Insert a batch of sheets records."""
    sql = """insert into splits ( split_set,  split,  coreid)
                         values (:split_set, :split, :coreid);"""
    insert_batch(database, sql, batch)


def select_split(
    *,
    database: DbPath,
    split_set: str,
    split: str,
    target_set: str,
    trait: str,
    limit: int = 0,
) -> list[dict]:
    """Select all records for a split_set/split combination."""
    sql = """select *
               from splits
               join angiosperms using (coreid)
               join images using (coreid)
               join targets using (coreid)
            """
    sql, params = build_select(
        sql,
        split_set=split_set,
        split=split,
        target_set=target_set,
        trait=trait,
        limit=limit,
    )
    return rows_as_dicts(database, sql, params)


def select_split_set_orders(database: DbPath, split_set: str) -> list[str]:
    """Get all of the phylogenetic orders for a split set."""
    sql = """select distinct order_
               from splits
               join angiosperms using (coreid)
              where split_set = ?
           order by order_"""
    with sqlite3.connect(database) as cxn:
        orders = [r[0] for r in cxn.execute(sql, (split_set,))]
    return orders


def select_all_split_sets(database: DbPath) -> list[str]:
    """Get all split set names in the database."""
    sql = """select distinct split_set from splits order by split_set"""
    with sqlite3.connect(database) as cxn:
        runs = [r[0] for r in cxn.execute(sql)]
    return runs


# ########### Test runs table ##########################################################


def create_tests_table(database: DbPath, drop: bool = False) -> None:
    """Create test runs table."""
    sql = """
        create table if not exists tests (
            coreid    text,
            test_set  text,
            split_set text,
            trait     text,
            target    real,
            pred      real
        );
        create unique index tests_idx on tests (coreid, test_set, trait);
        """
    create_table(database, sql, drop=drop)


def insert_tests(database: DbPath, batch: list, test_set: str, split_set: str) -> None:
    """Insert a batch of test set records."""
    sql = "delete from tests where test_set = ? and split_set = ?"
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, (test_set, split_set))
    sql = """insert into tests
                    ( coreid,  test_set,  split_set,  trait,  target,  pred)
             values (:coreid, :test_set, :split_set, :trait, :target, :pred);"""
    insert_batch(database, sql, batch)


def select_tests(database: DbPath, test_set: str, limit: int = 0) -> list[dict]:
    """Select all records for a test set."""
    sql = """select *
               from tests
               join angiosperms using (coreid)
               join images using (coreid)"""
    sql, params = build_select(sql, limit=limit, test_set=test_set)
    return rows_as_dicts(database, sql, params)


# ########### Inferences table #########################################################


def create_inferences_table(database: DbPath, drop: bool = False) -> None:
    """Save inference results."""
    sql = """
        create table if not exists inferences (
            coreid        text,
            inference_set text,
            trait         text,
            pred          real
        );
        create unique index inferences_idx on inferences (coreid, inference_set, trait);
        """
    create_table(database, sql, drop=drop)


def insert_inferences(database: DbPath, batch: list, inference_set: str) -> None:
    """Insert a batch of inference records."""
    sql = "delete from inferences where inference_set = ?"
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, (inference_set,))
    sql = """insert into inferences ( coreid,  inference_set,  trait,  pred)
                             values (:coreid, :inference_set, :trait, :pred);"""
    insert_batch(database, sql, batch)


def select_inferences(
    database: DbPath, inference_set: str, trait: str, limit: int = 0
) -> list[dict]:
    """Select all images."""
    sql = """ select * from inferences join angiosperms using (coreid) """
    sql, params = build_select(
        sql, inference_set=inference_set, trait=trait, limit=limit
    )
    return rows_as_dicts(database, sql, params)


def select_all_inference_sets(database: DbPath) -> list[str]:
    """Get all inference set names in the database."""
    sql = """
        select distinct inference_set
          from inferences
      order by inference_set"""
    with sqlite3.connect(database) as cxn:
        runs = [r[0] for r in cxn.execute(sql)]
    return runs
