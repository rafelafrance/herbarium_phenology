"""Given a CSV file of iDigBio records, download the images."""
import os
import socket
import sys
import warnings
from itertools import cycle
from itertools import groupby
from random import sample
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError

from herbarium.pylib import db


def sample_records(database, csv_dir, count=10_050, splits=10):
    """Get a broad sample of herbarium specimens."""
    sql = """
        with multiples as (
            select coreid, count(*) as n
              from angiosperms
          group by coreid
            having n > 1)
        select family, genus, coreid, accessuri
          from angiosperms
         where coreid not in (select coreid from multiples)
           and reproductivecondition <> ''
           and family <> ''
           and genus <> ''
           and accessuri <> ''
        """
    rows = db.rows_as_dicts(database, sql, [])
    rows = sample(rows, k=len(rows))

    row_groups = groupby(rows, key=lambda r: (r["family"], r["genus"]))
    groups = [list(g) for k, g in row_groups]
    groups = cycle(groups)

    uris = []
    for i in range(count):
        for j in range(1000):
            group = next(groups)
            if group:
                uris.append(group.pop(0))
                break
        else:
            raise ValueError("Ran out of groups.")

    splits = np.array_split(uris, splits)
    for i, data_set in enumerate(splits, 1):
        data = [(d["coreid"], d["accessURI"]) for d in data_set]
        path = csv_dir / f"uris_{i:02d}.csv"
        df = pd.DataFrame(data=data, columns=["coreid", "accessuri"])
        df.to_csv(path, index=False)


def download_images(
    csv_file,
    image_dir,
    error=None,
    url_column="accessuri",
    timeout=20,
):
    """Download iDigBio images out of a CSV file."""
    os.makedirs(image_dir, exist_ok=True)

    socket.setdefaulttimeout(timeout)

    error = error if error else sys.stderr

    df = pd.read_csv(csv_file, index_col="coreid", dtype=str)

    with open(error, "a") as err:
        for coreid, row in df.iterrows():
            path = image_dir / f"{coreid}.jpg"
            if path.exists():
                continue
            try:
                urlretrieve(row[url_column], path)
            except (HTTPError, URLError):
                err.write(f"Could not download: {row[url_column]}\n")
                err.flush()
                continue


def validate_images(image_dir, database, error=None, glob="*.jpg"):
    """Put valid image paths into a database."""
    error = error if error else sys.stderr
    images = []
    with warnings.catch_warnings():  # Turn off EXIF warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(error, "a") as err:
            for path in image_dir.glob(glob):
                image = None
                try:
                    image = Image.open(path)
                    width, height = image.size
                    images.append(
                        {
                            "coreid": path.stem,
                            "path": str(path).replace("../", ""),
                            "width": width,
                            "height": height,
                        }
                    )
                except UnidentifiedImageError:
                    err.write(f"Could not open: {path}\n")
                    err.flush()
                finally:
                    if image:
                        image.close()

    db.create_image_table(database, drop=True)
    db.insert_images(database, images)
