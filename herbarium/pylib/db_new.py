import sqlite3
from pathlib import Path
from typing import Union

DbPath = Union[Path, str]


def select(database: DbPath, sql, **kwargs):
    with sqlite3.connect(database) as cxn:
        cxn.row_factory = sqlite3.Row
        rows = cxn.execute(sql, dict(kwargs))
        return [dict(r) for r in rows]


def canned_select(database: DbPath, key: str, **kwargs):
    selects = {
        "orders": """select order_ from orders order by order_""",
        "all_target_sets": """
            select distinct target_set from targets order by target_set
            """,
        "images": """
            select *
            from   images
            join   angiosperms using (coreid)
            where  order_ in (select order_ from orders)
            """,
        "split": """
            select *
            from   splits
            join   angiosperms using (coreid)
            join   images using (coreid)
            join   targets using (coreid)
            where  split_set  = :split_set
            and    split      = :split
            and    target_set = :target_set
            and    trait      = :trait
            """,
        "split_set_orders": """
            select distinct order_
            from   splits
            join   angiosperms using (coreid)
            where  split_set = ?
            order by order_
            """,
        "all_split_sets": """
            select distinct split_set from splits order by split_set
            """,
        "tests": """
            select *
            from   tests
            join   angiosperms using (coreid)
            join   images using (coreid)
            where  test_set = :test_set
            """,
        "inferences": """
            select *
            from   inferences
            join   angiosperms using (coreid)
            where  inference_set = :inference_set
            and    trait         = :trait
            """,
        "pseudo_split": """
            select *
            from   images
            join   angiosperms using (coreid)
            where  order_ in (select order_ from orders)
            and    coreid not in (
                select coreid
                from   targets
                where  target_set = :target_set
                and    trait      = :trait
            )
            order by random()
            """,
        "all_inference_sets": """
            select distinct inference_set
            from   inferences
            order by inference_set
            """,
    }
    sql = selects[key]

    if kwargs.get("limit"):
        sql += " limit :limit"

    select(database, sql, **kwargs)


def canned_insert(database: DbPath, table, batch):
    inserts = {
        "targets": """
            insert into targets
                    ( coreid,  target_set,  filter_set,  trait,  target)
             values (:coreid, :target_set, :filter_set, :trait, :target);
             """,
        "images": """
            insert into images ( coreid,  path,  width,  height)
                        values (:coreid, :path, :width, :height);
            """,
        "splits": """
            insert into splits ( split_set,  split,  coreid)
                        values (:split_set, :split, :coreid);
            """,
        "tests": """
            insert into tests
                    ( coreid,  test_set,  split_set,  trait,  target,  pred)
             values (:coreid, :test_set, :split_set, :trait, :target, :pred);
            """,
        "inferences": """
            insert into inferences ( coreid,  inference_set,  trait,  pred)
                            values (:coreid, :inference_set, :trait, :pred);
            """,
    }
    with sqlite3.connect(database) as cxn:
        sql = inserts[table]
        cxn.executemany(sql, batch)


def canned_delete(database: DbPath, table: str, **kwargs):
    deletes = {
        "splits": """ delete from splits where split_set = :split_set """,
        "targets": """
            delete from targets where target_set = :target_set and trait = :trait
            """,
        "tests": """
            delete from tests where test_set = :test_set and split_set = :split_set
            """,
        "inferences": """
            delete from inferences where inference_set = :inference_set
            """,
    }
    sql = deletes[table]
    with sqlite3.connect(database) as cxn:
        cxn.execute(sql, dict(kwargs))


def create_table(database: DbPath, table="all", *, drop: bool = False):
    tables = {
        "targets": """
            create table if not exists targets (
                coreid     text,
                target_set text,
                filter_set text,
                trait      text,
                target     real
            );
            create unique index if not exists targets_idx
                on targets (target_set, trait, coreid);
            """,
        "images": """
            create table if not exists images (
                coreid text primary key,
                path   text unique,
                width  integer,
                height integer
            );
            """,
        "splits": """
            create table if not exists splits (
                split_set text,
                split     text,
                coreid    text
            );
            create unique index if not exists splits_idx on splits (split_set, coreid);
            """,
        "tests": """
            create table if not exists tests (
                coreid    text,
                test_set  text,
                split_set text,
                trait     text,
                target    real,
                pred      real
            );
            create unique index if not exists tests_idx
                on tests (test_set, trait, coreid);
            """,
        "inferences": """
            create table if not exists inferences (
                coreid        text,
                inference_set text,
                trait         text,
                pred          real
            );
            create unique index if not exists inferences_idx
                on inferences (inference_set, trait, coreid);
            """,
        "orders": """
            create table if not exists orders (
                order_ text
            );
            """,
        "angiosperms": """
            create table if not exists angiosperms (
                coreid                text,
                accessuri             text,
                basisofrecord         text,
                class_                text,
                continent             text,
                country               text,
                county                text,
                dynamicproperties     text,
                eventdate             text,
                family                text,
                fieldnotes            text,
                genus                 text,
                kingdom               text,
                locality              text,
                occurrenceremarks     text,
                order_                text,
                phylum                text,
                reproductivecondition text,
                scientificname        text,
                specificepithet       text,
                stateprovince         text,
                geopoint              text,
                filter_set            text,
                primary key (coreid)
            );
            create unique index if not exists angiosperms_coreid
                on angiosperms (coreid);
            """,
    }
    with sqlite3.connect(database) as cxn:
        for name, create in tables.items():
            if table == name or table == "all":
                if drop:
                    cxn.executescript(f"""drop table if exists {name};""")
                cxn.executescript(create)
