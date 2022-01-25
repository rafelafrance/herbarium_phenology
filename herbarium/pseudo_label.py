#!/usr/bin/env python3
"""Create a target dataset from inferred traits."""
import argparse
import textwrap
from pathlib import Path

from pylib import db
from pylib import validate_args as val


def label(args):
    """Create the target dataset."""
    all_inf = db.select_inferences(args.database, args.inference_set)

    batch = []

    for inf in all_inf:
        if inf["pred"] <= args.min_threshold:
            target = 0.0
        elif inf["pred"] >= args.max_threshold:
            target = 1.0
        else:
            continue

        batch.append(
            {
                "coreid": inf["coreid"],
                "target_set": args.target_set,
                "trait": inf["trait"],
                "target": target,
            }
        )

    db.insert_targets(args.database, batch, args.target_set)


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

    val.validate_inference_set(args)

    return args


def main():
    """Infer traits."""
    args = parse_args()
    label(args)


if __name__ == "__main__":
    main()
