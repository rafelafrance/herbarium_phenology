#!/usr/bin/env python3
"""Create a target dataset from inferred traits."""
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib import validate_args as val
from pylib.const import ALL_TRAITS


def label(args):
    """Create the target dataset."""

    batch = extend_target_set(
        args.database, args.target_set, args.base_target_set, args.trait
    )
    ids = {r["coreid"] for r in batch}

    for row in db.select_inferences(args.database, args.inference_set, args.trait):
        if row["coreid"] in ids:
            continue

        if row["pred"] <= args.min_threshold:
            target = 0.0
        elif row["pred"] >= args.max_threshold:
            target = 1.0
        else:
            continue

        ids.add(row["coreid"])

        batch.append(
            {
                "coreid": row["coreid"],
                "target_set": args.target_set,
                "source_set": f"pseudo_{args.inference_set}",
                "trait": row["trait"],
                "target": target,
            }
        )

    db.insert_targets(args.database, batch, args.target_set, args.trait)


def extend_target_set(database, target_set, base_target_set, trait) -> list[dict]:
    """Start with this as the base target set."""
    batch = []
    if base_target_set:
        sql = """
            select *
              from targets
              join images using (coreid)
             where target_set = ?
               and trait = ?"""
        batch = db.rows_as_dicts(database, sql, [base_target_set, trait])
        for row in batch:
            source_set = row["source_set"] if row["source_set"] else base_target_set
            row["source_set"] = source_set
            row["target_set"] = target_set
    return batch


def parse_args():
    """Process command-line arguments."""
    description = """Create a target dataset from an inference set."""
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
        "--target-set",
        metavar="NAME",
        required=True,
        help="""Give the target dataset this name.""",
    )

    arg_parser.add_argument(
        "--inference-set",
        metavar="NAME",
        required=True,
        help="""Use this inference set for creating the target set.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=ALL_TRAITS,
        required=True,
        help="""Which trait to infer.""",
    )

    arg_parser.add_argument(
        "--base-target-set",
        metavar="NAME",
        help="""Start with this target set. Add target records from this set before
            adding any records from the --inference-set.""",
    )

    arg_parser.add_argument(
        "--min-threshold",
        type=float,
        metavar="FLOAT",
        default=0.1,
        help="""When the inference value for the trait is below this the trait is
            considered to be absent. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-threshold",
        type=float,
        metavar="FLOAT",
        default=0.9,
        help="""When the inference value for the trait is above this the trait is
            considered to be present. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    val.validate_inference_set(args.database, args.inference_set)
    if args.base_target_set:
        val.validate_target_set(args.database, args.base_target_set)

    return args


def main():
    """Infer traits."""
    args = parse_args()
    label(args)


if __name__ == "__main__":
    main()
