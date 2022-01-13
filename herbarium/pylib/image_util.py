"""Given download the images in the iDigBio database."""
import os
import socket
import sys
import warnings
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import DecompressionBombWarning
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm

from herbarium.pylib import db
from herbarium.pylib.herbarium_old_dataset import HerbariumOldDataset
from herbarium.pylib.idigbio_load import TRAITS

# Make a few attempts to download a page
ATTEMPTS = 3

# Set a timeout for requests
TIMEOUT = 10
socket.setdefaulttimeout(TIMEOUT)

# The column that holds the image URL
COLUMN = "accessuri"

# Catch these errors during downloads and image validation
ERRORS = (
    AttributeError,
    BufferError,
    EOFError,
    IndexError,
    IOError,
    OSError,
    RuntimeError,
    SyntaxError,
    TypeError,
)


def sample_records(database, csv_dir, splits=8, limit=100):
    """Get a broad sample of herbarium specimens."""
    orders = db.select_all_orders(database)
    rows = []
    for order in tqdm(orders):
        for trait in TRAITS:
            sql = f"""
                select coreid, accessuri
                  from angiosperms
                where {trait} = 1
                  and order_ = ?
                  and coreid not in (select coreid from images)
                  and accessuri not like '%harvard%'
                  and accessuri not like '%uconn%'
             order by random()
            """
            sql, _ = db.build_select(sql, limit=limit)
            rows += db.rows_as_dicts(database, sql, [order, limit])

    for i, array in enumerate(np.array_split(rows, splits)):
        df = pd.DataFrame(array.tolist())
        df.to_csv(csv_dir / f"uris_{i}.csv", index=False)


def download_images(csv_file, image_dir, error=None):
    """Download iDigBio images out of a CSV file."""
    os.makedirs(image_dir, exist_ok=True)

    df = pd.read_csv(csv_file, index_col="coreid", dtype=str)

    with open(error, "a") if error else sys.stderr as err:
        for coreid, row in df.iterrows():
            path = image_dir / f"{coreid}.jpg"
            if path.exists():
                continue

            for _ in range(ATTEMPTS):
                try:
                    urlretrieve(row[COLUMN], path)
                    break
                except ERRORS:
                    pass  # Gets handled in the for loop's else clause
            else:
                print(f"Could not download: {row[COLUMN]}", file=err, flush=True)


def validate_images(image_dir, database, error, glob="*.jpg"):
    """Put valid image paths into the database."""
    images = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
        warnings.filterwarnings("error", category=DecompressionBombWarning)
        with open(error, "a") if error else sys.stderr as err:
            for path in tqdm(image_dir.glob(glob)):
                image = None
                try:
                    image = Image.open(path)
                    image.verify()
                    width, height = image.size
                    _ = io.imread(path)
                    images.append(
                        {
                            "coreid": path.stem,
                            "path": str(path).replace("../", ""),
                            "width": width,
                            "height": height,
                        }
                    )
                except ERRORS as e:  # Image isn't added to DB
                    err.write(f"Bad image: {path} {e}\n")
                    err.flush()
                finally:
                    if image:
                        image.close()

    db.create_images_table(database, drop=True)
    db.insert_images(database, images)


def get_image_norm(database, classifier, split_run, batch_size=16, num_workers=4):
    """Get the mean and standard deviation of the image channels."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = db.select_split(database, split_run, split="train")
    split = HerbariumOldDataset(data, classifier)
    loader = DataLoader(split, batch_size=batch_size, num_workers=num_workers)

    # TODO: Has bad round-off error according to Numerical Recipes in C, 2d ed. p 613
    sum_, sq_sum, count = 0.0, 0.0, 0

    for images, _ in tqdm(loader):
        images = images.to(device)
        sum_ += torch.mean(images, dim=[0, 2, 3])
        sq_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        count += 1

    mean = sum_ / count
    std = (sq_sum / count - mean ** 2) ** 0.5
    return mean, std
