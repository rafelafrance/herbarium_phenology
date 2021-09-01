#!/usr/bin/python3

# Copyright (C) 2021 Brian J. Stucky
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Downloads images from a set of iDigBio search results, using random names for
# the downloaded files, and generates a CSV file mapping downloaded file names
# to source URIs, iDigBio core IDs, and scientific names.
#
# @author: Brian J. Stucky
#


import os.path
import csv
from argparse import ArgumentParser
import time
import sqlite3
from imglib import getImageSize, mtDownload, ImageDownloadWorkerResult


def getArgParser():
    argp = ArgumentParser(
        description='Downloads images from a set of SQLite database query '
        'results.  All downloaded image files will be converted to JPEG '
        'format, if needed.'
    )
    argp.add_argument(
        '-d', '--database', type=str, required=True, help='A SQLite database.'
    )
    argp.add_argument(
        '-q', '--queryfile', type=str, required=True, help='A file containing '
        'a single SQL query to use for retrieving images.'
    )
    argp.add_argument(
        '-i', '--idcol', type=str, required=False, default='coreid',
        help='The name of the results column containing a unique record ID, '
        'which will be used for the donwloaded image file names '
        '(default: "coreid").'
    )
    argp.add_argument(
        '-u', '--uricol', type=str, required=False, default='imgURI',
        help='The name of the results column containing the image URIs '
        '(default: "imgURI").'
    )
    argp.add_argument(
        '-o', '--fileout', type=str, required=False,
        default='downloaded_files.csv', help='The path to an output file for '
        'writing image file information (default: "downloaded_files.csv").'
    )
    argp.add_argument(
        '-g', '--imgdir', type=str, required=False, default='raw_images',
        help='The path to a directory in which to place the downloaded images '
        '(default: "raw_images"). The directory will be created if it does '
        'not already exist.'
    )
    argp.add_argument(
        '-t', '--timeout', type=int, required=False, default=20, help='The '
        'number of seconds to wait before HTTP connect or read timeout '
        'failures (default: 20).'
    )
    argp.add_argument(
        '-r', '--threads', type=int, required=False, default=20, help='The '
        'maximum number of threads to use for concurrent downloads '
        '(default: 20).'
    )

    return argp

def checkPaths(args):
    if os.path.exists(args.imgdir) and not(os.path.isdir(args.imgdir)):
        raise IOError(
            'A file with the same name as the output image directory, '
            '"{0}", already exists.'.format(args.imgdir)
        )

    if not(os.path.exists(args.imgdir)):
        os.mkdir(args.imgdir)

def getDownloadRequests(db, qf, id_col, uri_col, img_folder, dl_fpath):
    """
    Generates a list of download request (core ID, URI, other_fields) tuples.

    db: A SQLite database.
    qf: A SQL query file.
    id_col: The ID column name.
    uri_col: The image URI column name.
    img_folder: The location of the downloaded images.
    dl_fpath: An output file of downloaded file information.
    """
    # Get the set of successfully downloaded core IDs.
    dl_coreids = set()
    if os.path.isfile(dl_fpath):
        with open(dl_fpath) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                fpath = os.path.join(img_folder, row['file'])
                if os.path.isfile(fpath):
                    dl_coreids.add(row['coreid'])
                else:
                    raise Exception(
                        'Downloaded image file not found: {0}.'.format(
                            row['file']
                        )
                    )

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    with open(qf) as fin:
        query = fin.read().strip()

    if query == '':
        raise Exception(f'No SQL query was found in {qf}.')

    downloads = []
    for row in c.execute(query):
        coreid = row[id_col]

        if coreid not in dl_coreids:
            imguri = row[uri_col]
            if imguri is not None and imguri != '':
                other_vals = dict(row)
                del other_vals[id_col]
                del other_vals[uri_col]

                downloads.append((coreid, imguri, other_vals))
            else:
                print(f'WARNING: No image URI for coreid {coreid}.')
        else:
            print('Skipping extant image for {0}...'.format(coreid))

    return downloads, list(other_vals.keys())


argp = getArgParser()
args = argp.parse_args()

try:
    checkPaths(args)
except Exception as err:
    exit('\nERROR: {0}\n'.format(err))

downloads, other_colnames = getDownloadRequests(
    args.database, args.queryfile, args.idcol, args.uricol, args.imgdir,
    args.fileout
)
if len(downloads) == 0:
    exit()

# Generate the file name for the failure log.
logfn = 'fail_log-{0}.csv'.format(
    time.strftime('%Y%b%d-%H:%M:%S', time.localtime())
)

if os.path.isfile(args.fileout):
    dl_file_size = os.path.getsize(args.fileout)
else:
    dl_file_size = 0

# Process the download requests.
with open(args.fileout, 'a') as fout, open(logfn, 'w') as logf:
    writer = csv.DictWriter(
        fout,
        ['file', 'imgsize', 'bytes', 'coreid', 'uri', 'time'] + other_colnames
    )
    if dl_file_size == 0:
        writer.writeheader()

    faillog = csv.DictWriter(
        logf, ['coreid', 'uri', 'time', 'reason'] + other_colnames
    )
    faillog.writeheader()

    for result in mtDownload(
        downloads, args.imgdir, args.timeout, args.threads
    ):
        outrow = {}

        outrow['coreid'] = result.identifier
        outrow['uri'] = result.uri
        outrow['time'] = result.timestamp

        for colname in other_colnames:
            outrow[colname] = result.other_data[colname]

        if result.result == ImageDownloadWorkerResult.SUCCESS:
            imgpath = os.path.join(args.imgdir, result.localpath)
            outrow['file'] = result.localpath
            outrow['imgsize'] = getImageSize(imgpath)
            outrow['bytes'] = os.stat(imgpath).st_size
            writer.writerow(outrow)
        elif result.result == ImageDownloadWorkerResult.DOWNLOAD_FAIL:
            outrow['reason'] = result.fail_reason
            faillog.writerow(outrow)

