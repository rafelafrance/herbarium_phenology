"""Download the images in the iDigBio database."""
import os
import socket
import sys
import warnings
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import DecompressionBombWarning
from skimage import io
from tqdm import tqdm

from .. import db
from ..consts import TRAITS
from .idigbio_consts import COLUMN

# Make a few attempts to download a page
ATTEMPTS = 3

# Set a timeout for requests
TIMEOUT = 10
socket.setdefaulttimeout(TIMEOUT)

# Catch these errors during downloads and image validation
# Python error handling sucks!
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
    ValueError,
    DecompressionBombWarning,
)


def sample_records(database, csv_dir, splits=8, limit=100):
    """Get a broad sample of utils specimens."""
    orders = db.canned_select(database, "orders", one_column=True)
    rows = []
    for order in tqdm(orders):
        for trait in TRAITS:
            sql = """
                select coreid, accessuri
                  from angiosperms
                  join targets using (coreid)
                where trait = :trait
                  and order_ = :order
                  and coreid not in (select coreid from images)
             order by random()
            """
            rows += db.select(database, sql, trait=trait, order=order, limit=limit)

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


def validate_images(image_dir, database, error, glob="*.jpg", every=100):
    """Put valid image paths into the database."""
    db.create_table(database, "images")

    sql = """select * from images"""
    existing = {r["coreid"] for r in db.select(database, sql)}
    new_paths = {p for p in image_dir.glob(glob) if p.stem not in existing}
    images = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
        warnings.filterwarnings("error", category=DecompressionBombWarning)
        with open(error, "a") if error else sys.stderr as err:
            for path in tqdm(new_paths):
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

                if len(images) % every == 0:
                    db.canned_insert(database, "images", images)
                    images = []

    db.canned_insert(database, "images", images)


# def get_image_norm(database, classifier, split_set, batch_size=16, num_workers=4):
#     """Get the mean and standard deviation of the image channels."""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     data = db.select_split(database, split_set, split="train")
#     split = HerbariumDataset(data, classifier)
#     loader = DataLoader(split, batch_size=batch_size, num_workers=num_workers)
#
#     # TODO: Has bad round-off error according to Numerical Recipes in C, 2d ed. p 613
#     sum_, sq_sum, count = 0.0, 0.0, 0
#
#     for images, _ in tqdm(loader):
#         images = images.to(device)
#         sum_ += torch.mean(images, dim=[0, 2, 3])
#         sq_sum += torch.mean(images ** 2, dim=[0, 2, 3])
#         count += 1
#
#     mean = sum_ / count
#     std = (sq_sum / count - mean ** 2) ** 0.5
#     return mean, std
