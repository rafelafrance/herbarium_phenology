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
            where.append("{key} = ?")
            params.append(value)

    sql += (" where " + " and ".join(where)) if where else ""
    return sql, params


def rows_as_dicts(database: DbPath, sql: str, params: list):
    """Convert the SQL execute cursor to a list of dicts."""
    with sqlite3.connect(database) as cxn:
        cxn.row_factory = sqlite3.Row
        rows = [dict(r) for r in cxn.execute(sql, params)]
    return rows


def insert_batch(database: DbPath, sql: str, batch: list) -> None:
    """Insert a batch of sheets records."""
    if batch:
        with sqlite3.connect(database) as cxn:
            cxn.executemany(sql, batch)


def create_table(database: DbPath, sql: str, table: str, *, drop: bool = False) -> None:
    """Create a table with paths to the valid herbarium sheet images."""
    with sqlite3.connect(database) as cxn:
        if drop:
            cxn.executescript(f"""drop table if exists {table};""")

        cxn.executescript(sql)


# ########### Image tables ##########################################################


def create_image_table(database: DbPath, drop: bool = False) -> None:
    """Create a table with paths to the valid herbarium sheet images."""
    sql = """
        create table if not exists images (
            coreid   text primary key,
            path     text unique,
            width    integer,
            height   integer
        );
        """
    create_table(database, sql, "images", drop=drop)


def insert_images(database: DbPath, batch: list) -> None:
    """Insert a batch of sheets records."""
    sql = """insert into images ( coreid,  path,  width,  height)
                         values (:coreid, :path, :width, :height);"""
    insert_batch(database, sql, batch)


def select_images(database: DbPath, *, limit: int = 0) -> list[dict]:
    """Get herbarium sheet image data."""
    sql = """select * from images"""
    sql, params = build_select(sql, limit=limit)
    return rows_as_dicts(database, sql, params)
