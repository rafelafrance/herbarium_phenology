"""Given a CSV file of iDigBio records, download the images."""
import os
import socket
import sqlite3
import sys
import warnings
from urllib.request import urlretrieve

import pandas as pd
import torch
from PIL import Image
from PIL.Image import DecompressionBombWarning
from skimage import io
from torchvision import transforms as xfm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

from phenology.pylib import db
from phenology.pylib.idigbio_load import FLAGS

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
                    # Gets handled in the for loop's else clause
                    pass
            else:
                print(f"Could not download: {row[COLUMN]}", file=err, flush=True)


def validate_images(image_dir, database, error=None, glob="*.jpg"):
    """Put valid image paths into a database."""
    error = error if error else sys.stderr
    images = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)   # No EXIF warnings
        warnings.filterwarnings("error", category=DecompressionBombWarning)
        with open(error, "a") as err:
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
                except Exception as e:  # pylint: disable=broad-except
                    # Image doesn't get added to DB
                    err.write(f"Bad image: {path} {e}\n")
                    err.flush()
                finally:
                    if image:
                        image.close()

    db.create_image_table(database, drop=True)
    db.insert_images(database, images)


def get_image_norm(image_dir, batch_size=16, size=None):
    """Get the mean and standard deviation of the image channels."""
    # TODO: has bad round-off error according to Numerical Recipes in C, 2d ed. p 613
    size = size if size else (224, 224)
    transforms = xfm.Compose([xfm.Resize(size), xfm.ToTensor()])
    dataset = ImageFolder(str(image_dir), transform=transforms)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    sum_, sq_sum, count = 0.0, 0.0, 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
        for images, _ in tqdm(loader):
            sum_ += torch.mean(images, dim=[0, 2, 3])
            sq_sum += torch.mean(images**2, dim=[0, 2, 3])
            count += 1
    mean = sum_ / count
    std = (sq_sum / count - mean**2)**0.5
    return mean, std
