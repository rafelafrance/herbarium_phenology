"""Misc. utilities that don't fit in any other module."""

from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp


@contextmanager
def make_temp_dir(where=None, prefix=None, keep=False):
    """Handle creation and deletion of temporary directory in/outside of /tmp."""
    temp_dir = mkdtemp(prefix=prefix, dir=where)
    try:
        yield temp_dir
    finally:
        if not (keep and where):
            rmtree(temp_dir, ignore_errors=True)


def collate_fn(batch):
    """Turn batches into tuples."""
    return tuple(zip(*batch))
