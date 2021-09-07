"""Misc. utilities that don't fit in any other module."""

from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp


@contextmanager
def make_temp_dir(where=None, prefix=None, keep=False):
    """Handle creation and deletion of temporary directory in/outside of /tmp.

    where:  Path/string where to put the temp dir. If none then use /tmp.
    prefix: The new directory's a name prefix. So you can identify it easily.
    keep:   Should we avoid deleting the temp dir when done? Used for debugging.
    """
    temp_dir = mkdtemp(prefix=prefix, dir=where)
    try:
        yield temp_dir
    finally:
        if not (keep and where):
            rmtree(temp_dir, ignore_errors=True)


def collate_fn(batch):
    """Turn batches into tuples."""
    return tuple(zip(*batch))
