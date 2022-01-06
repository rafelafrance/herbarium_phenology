"""Utilities for angiosperm.sqlite databases."""
import sqlite3
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
    with sqlite3.connect(database) as cxn:
        cxn.row_factory = sqlite3.Row
        rows = [dict(r) for r in cxn.execute(sql, params)]
    return rows


def insert_batch(database: DbPath, sql: str, batch: list) -> None:
    """Insert a batch of records."""
    if batch:
        with sqlite3.connect(database) as cxn:
            cxn.executemany(sql, batch)


def create_table(database: DbPath, sql: str, table: str, *, drop: bool = False) -> None:
    """Create a table."""
    with sqlite3.connect(database) as cxn:
        if drop:
            cxn.executescript(f"""drop table if exists {table};""")

        cxn.executescript(sql)


# ########### Image table ##########################################################


def create_images_table(database: DbPath, drop: bool = False) -> None:
    """Create a table with paths to the valid herbarium sheet images."""
    sql = """
        create table if not exists images (
            coreid    text primary key,
            path      text unique,
            width     integer,
            height    integer
        );
        """
    create_table(database, sql, "images", drop=drop)


def insert_images(database: DbPath, batch: list) -> None:
    """Insert a batch of sheets records."""
    sql = """insert into images ( coreid,  path,  width,  height)
                         values (:coreid, :path, :width, :height);"""
    insert_batch(database, sql, batch)


def select_images(database: DbPath, limit: int = 0) -> list[dict]:
    """Select all images."""
    sql = """select *
               from images
               join angiosperms using (coreid)"""
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
            split_run text,
            split     text,
            coreid    text
        );
        """
    create_table(database, sql, "splits", drop=drop)


def insert_splits(database: DbPath, batch: list) -> None:
    """Insert a batch of sheets records."""
    sql = """insert into splits ( split_run,  split,  coreid)
                         values (:split_run, :split, :coreid);"""
    insert_batch(database, sql, batch)


def select_split(
    database: DbPath, split_run: str, split: str, limit: int = 0
) -> list[dict]:
    """Select all records for a split_run/split combination."""
    sql = """select *
               from splits
               join angiosperms using (coreid)
               join images using (coreid)"""
    sql, params = build_select(sql, limit=limit, split_run=split_run, split=split)
    return rows_as_dicts(database, sql, params)


def select_all_orders(database: DbPath) -> list[str]:
    """Get all orders with images."""
    sql = """select distinct order_ from angiosperms order by order_"""
    with sqlite3.connect(database) as cxn:
        orders = [r[0] for r in cxn.execute(sql)]
    return orders


def select_orders(database: DbPath, split_run: str) -> list[str]:
    """Get all of the phylogenetic orders for a split run."""
    sql = """select distinct order_
               from splits
               join angiosperms using (coreid)
              where split_run = ?
           order by order_"""
    with sqlite3.connect(database) as cxn:
        orders = [r[0] for r in cxn.execute(sql, (split_run,))]
    return orders


# ########### Test runs table ##########################################################


def create_test_runs_table(database: DbPath, drop: bool = False) -> None:
    """Create train/validation/test splits of the data.

    This is so I don't wind up training on my test data. Because an image can belong
    to multiple classes I need to be careful that I don't add any core IDs in the
    test split to the training/validation splits.
    """
    sql = """
        create table if not exists test_runs (
            coreid    text,
            test_run  text,
            split_run text,
            trait     text,
            true      real,
            pred      real
        );
        """
    create_table(database, sql, "test_runs", drop=drop)


def insert_test_runs(
    database: DbPath, batch: list, test_run: str, split_run: str
) -> None:
    """Insert a batch of sheets records."""
    sql = "delete from test_runs where test_run = ? and split_run = ?"
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, (test_run, split_run))
    sql = """insert into test_runs
                    ( coreid,  test_run,  split_run,  trait,  true,  pred)
             values (:coreid, :test_run, :split_run, :trait, :true, :pred);"""
    insert_batch(database, sql, batch)


# ########### Inferences table #########################################################


def create_inferences_table(database: DbPath, drop: bool = False) -> None:
    """Create train/validation/test splits of the data.

    This is so I don't wind up training on my test data. Because an image can belong
    to multiple classes I need to be careful that I don't add any core IDs in the
    test split to the training/validation splits.
    """
    sql = """
        create table if not exists inferences (
            coreid        text,
            inference_run text,
            trait         text,
            pred          real
        );
        """
    create_table(database, sql, "test_runs", drop=drop)


def insert_inferences(database: DbPath, batch: list, inference_run: str) -> None:
    """Insert a batch of sheets records."""
    sql = "delete from inferences where inference_run = ?"
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, (inference_run,))
    sql = """insert into inferences ( coreid,  inference_run,  trait,  pred)
                             values (:coreid, :inference_run, :trait, :pred);"""
    insert_batch(database, sql, batch)
