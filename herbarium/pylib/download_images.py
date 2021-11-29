"""Given a CSV file of iDigBio records, download the images."""
import os
import socket
import sqlite3
import sys
import warnings
from urllib.request import urlretrieve

import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError

from herbarium.pylib import db
from herbarium.pylib.idigbio_load import FLAGS

# Make a few attempts to download a page
ATTEMPTS = 3

# Set a timeout for requests
TIMEOUT = 10
socket.setdefaulttimeout(TIMEOUT)

# The column that holds the image URL
COLUMN = "accessuri"


def sample_records(database, csv_dir, limit=1000):
    """Get a broad sample of herbarium specimens."""
    sql_template = """
        select coreid, accessuri
          from angiosperms
         where {} = 1
      order by random()
         limit {};
        """
    queries = {f"uris_{f}.csv": sql_template.format(f, limit) for f in FLAGS}

    with sqlite3.connect(database) as cxn:
        for file_name, query in queries.items():
            df = pd.read_sql(query, cxn)
            df.to_csv(csv_dir / file_name, index=False)


def download_images(csv_file, image_dir, error=None):
    """Download iDigBio images out of a CSV file."""
    os.makedirs(image_dir, exist_ok=True)

    error = error if error else sys.stderr

    df = pd.read_csv(csv_file, index_col="coreid", dtype=str)

    with open(error, "a") as err:
        for coreid, row in df.iterrows():
            path = image_dir / f"{coreid}.jpg"
            if path.exists():
                continue

            for attempt in range(ATTEMPTS):
                try:
                    urlretrieve(row[COLUMN], path)
                    break
                except Exception:  # pylint: disable=broad-except
                    # And catching 8+ different exceptions isn't ideal either
                    pass
            else:
                print(f"Could not download: {row[COLUMN]}", file=err, flush=True)


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
