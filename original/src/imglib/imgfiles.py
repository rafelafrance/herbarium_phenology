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

#
# Utility classes and functions for working with image files at the file system
# level, including facilities for parallel processing.
#
# @author: Brian J. Stucky
#


import os
import os.path
import subprocess
import time
import imghdr
from PIL import Image
import threading as mt
from queue import Queue as MTQueue
import string
import urllib.request as req
import urllib.parse
import shutil


IMG_EXTS = (
    '.jpg', '.jpeg', '.JPG', '.JPEG', '.tif', '.tiff', '.TIF', '.TIFF',
    '.png', '.PNG'
)


def findImageFiles(dirname, sort=True, cs_sort=False, inspect=False):
    """
    Returns a list of all raster image files in a specified directory.  The
    list elements are full paths (either relative or absolute, depending on the
    value of "dirname").  If "inspect" is True, the actual contents of each
    file will be inspected to determine if the file is an image file.  If
    False, file name extensions will be used to determine file type.  If
    possible, "inspect" should be False for processing large image libraries
    because it is much faster.  On a typical laptop running GNU/Linux with
    Python 2.7, file contents inspection takes ~2.37 times longer than file
    name extension filtering.  Oddly, the performance gap is even worse with
    Python 3.5, where file contents inspection takes ~2.89 times longer.

    sort: If True, the file names are returned in ascending alphanumeric sorted
        order.
    cs_sort: If True, the sort on file names will be case-sensitive.
    inspect: If True, inspect file contents to determine file types; otherwise,
        use file name extensions to determine file types.
    """
    fnames = []

    for fname in os.listdir(dirname):
        fpath = os.path.join(dirname, fname)
        if os.path.isfile(fpath):
            if inspect:
                if imghdr.what(fpath) is not None:
                    fnames.append(fpath)
            elif os.path.splitext(fpath)[1] in IMG_EXTS:
                fnames.append(fpath)

    if sort:
        if cs_sort:
            fnames.sort()
        else:
            fnames.sort(key=str.lower)

    return fnames

def getImageSize(fpath):
    """
    Returns the size of an image file in pixels.  The size is returned as the
    tuple (width, height).
    """
    img = Image.open(fpath)
    size = img.size
    img.close()
    
    return size

def convertToJPEG(fpath):
    """
    Takes a raw image file and converts it to JPEG format, if needed, and adds
    the ".jpg" extension to the file name, if needed.  Returns the new file
    name.  To support JPEG 2000, this function currently uses a call to the
    external "file" utility for file type detection and uses GraphicsMagick for
    image format conversion.
    """
    cmdargs = ['file', '-b', fpath]

    result = subprocess.run(
        cmdargs, check=True, stdout=subprocess.PIPE, universal_newlines=True
    )
    resultstr = result.stdout

    newname = fpath + '.jpg'

    if resultstr.startswith('JPEG image data'):
        os.rename(fpath, newname)
    elif resultstr.split()[0] in ('JPEG', 'PNG', 'TIFF', 'GIF'):
        #cmdstr = '/usr/local/bin/gm convert -quality 97% {0} {1}'.format(fpath, newname)
        cmdstr = 'convert -quality 97% {0} {1}'.format(fpath, newname)
        subprocess.run(cmdstr, check=True, shell=True)
        os.unlink(fpath)
    else:
        raise Exception(
            'Unsupported image format encountered in file {0}: "{1}".'.format(
                fpath, result
            )
        )

    return newname



class MTListReader:
    """
    Implements a thread-safe wrapper for sequentially accessing items in a list
    or other integer-indexible object that supports len().  Includes an error
    flag, self.error, to indicate whether processing should continue.  If the
    flag is set, MTListReader.STOPVAL will be returned by all subsequent calls
    to nextItem().
    """
    STOPVAL = None

    def __init__(self, data):
        """
        data: An integer-indexible object that supports len().
        """
        self.data = data
        self.index = 0
        self.lock = mt.Lock()
        self.error = mt.Event()

    def __len__(self):
        return len(self.data)

    def nextItem(self):
        """
        Returns the next item in the sequence along with its index as the tuple
        (index, item).  At the end of the sequence, or if the error event is
        set, returns None.
        """
        with self.lock:
            retval = self.STOPVAL
            if self.index < len(self.data) and not(self.error.is_set()):
                retval = (self.index, self.data[self.index])
                self.index += 1

        return retval


class ImageDownloadWorkerResult:
    """
    This is a simple struct-like class that ImageDownloadWorker instances use
    to report the results of download tasks.
    """
    SUCCESS = 0
    DOWNLOAD_FAIL = 1
    ERROR = 2

    def __init__(self):
        # Indicates the overall result of the download operation.
        self.result = None
        # An (optional) identifier for the download operation, separate from
        # the URI.
        self.identifier = None
        # The remote resource URI.
        self.uri = None
        # The path for the local copy of the resource, if the download was
        # successful.
        self.localpath = None
        # The time at which the download was completed.
        self.timestamp = None
        # The reason for a download failure, if known.
        self.fail_reason = None
        # Any exception generated by the download operation.
        self.exception = None
        # Other data provided with the download request.
        self.other_data = None


class ImageDownloadWorker(mt.Thread):
    """
    A subclass of multithreading.Thread designed for downloading image files.

    I investigated three alternative implementations for HTTP downloads:
    requests, urllib, and wget (using subprocesses).  I tested the performance
    of all three using a sample of 100 image URIs from iDigBio.  For the
    requests and urllib implementations, I also experimented with different
    chunk/buffer sizes.  For all implementations, I experimented with different
    numbers of threads: 1, 10, 20, and 40.  I found that urllib consistently
    produced the best results, with the requests and wget implementations
    performing similarly, but neither as fast as urllib.  Larger chunk sizes (1
    KB or greater) significantly improved the performance of requests, but even
    going up to 1 MB chunks, it still was slower than urllib.  Although the
    documentation suggests that None should work as a chunk size, the downloads
    always hung when I tried this.  The default urllib buffer size (16*1024)
    seemed to work quite well, but I found some evidence that performance
    improved very slightly when I increased this to 1 MB, so that is the
    default.  On the test set, 10, 20, and 40 threads all performed similarly,
    but 20 appeared to be the optimal number for the test set, so that is the
    default.  Here are sample benchmarks from server-class hardware for urllib
    with a 1 MB buffer size (best time of 3 trials for each test):
         1 thread : 2 m, 19.16 s
        10 threads: 0 m, 20.10 s
        20 threads: 0 m, 18.82 s
        40 threads: 0 m, 20.37 s
    """
    STOPVAL = MTListReader.STOPVAL
    FNCHARS = string.digits + string.ascii_letters

    def __init__(
        self, downloads, outputq, outputdir, timeout=20, update_console=True
    ):
        """
        downloads: An MTListReader for retrieving download requests.
        outputq: A Queue for reporting back image download results.
        outputdir: A directory in which to save downloaded images.
        timeout: Seconds to wait before connect or read timeouts.
        update_console: If True, send download updates to the console.
        """
        super(ImageDownloadWorker, self).__init__()

        self.downloads = downloads
        self.outputq = outputq
        self.outputdir = outputdir
        self.timeout = timeout
        self.update_console = update_console

    def _isImageIncomplete(self, fpath):
        """
        Tries to determine if an image download completely failed or was only
        partially successful.  Returns True if the image download failed, False
        otherwise.  Because the default Pillow binary does not include JPEG
        2000 support, partial downloads are not detected for JPEG 2000 files.
        """
        if not(os.path.exists(fpath)):
            return True

        if os.stat(fpath).st_size == 0:
            return True

        try:
            img = Image.open(fpath)
        except IOError:
            return True

        if img.format != 'JPEG2000':
            try:
                pxdata = img.load()
            except IOError:
                return True

        return False

    def run(self):
        for index, item in iter(self.downloads.nextItem, self.STOPVAL):
            identifier, uri, other_data = item

            result = ImageDownloadWorkerResult()
            result.uri = uri
            result.identifier = identifier
            result.other_data = other_data
            if self.update_console:
                print(
                    '({1:.1f}%) Downloading {0}...'.format(
                        uri, ((index + 1) / len(self.downloads)) * 100
                    )
                )

            try:
                imgfpath = os.path.join(self.outputdir, identifier)
                reqerr = None

                parts = urllib.parse.urlsplit(uri)
                parts = parts._replace(
                    path=urllib.parse.quote_plus(parts.path, safe='/')
                )
                parts = parts._replace(
                    query=urllib.parse.quote_plus(parts.query, safe='=&')
                )
                safe_uri = urllib.parse.urlunsplit(parts)
                try:
                    httpr = req.urlopen(safe_uri, timeout=self.timeout)
                    with open(imgfpath, 'wb') as imgout:
                        shutil.copyfileobj(httpr, imgout, 1024*1024)
                except OSError as err:
                    # Note that urllib.error.URLError and all low-level socket
                    # exceptions are subclasses of OSError.
                    reqerr = err

                result.timestamp = time.strftime(
                    '%Y%b%d-%H:%M:%S', time.localtime()
                )

                # Check if the download succeeded.
                if reqerr is None and not(self._isImageIncomplete(imgfpath)):
                    newfpath = convertToJPEG(imgfpath)
                    result.result = ImageDownloadWorkerResult.SUCCESS
                    result.localpath = os.path.basename(newfpath)
                else:
                    result.result = ImageDownloadWorkerResult.DOWNLOAD_FAIL
                    if reqerr is not None:
                        result.fail_reason = str(reqerr)

                    if os.path.exists(imgfpath):
                        os.unlink(imgfpath)

                self.outputq.put(result)
            except Exception as err:
                result.result = ImageDownloadWorkerResult.ERROR
                result.exception = err
                self.outputq.put(result)


def mtDownload(downloads, imgdir, timeout=20, maxthread_cnt=20):
    """
    Initiates multithreaded downloading of a list of download requests.  This
    function returns a generator object that can be used to iterate over all
    download request results.

    downloads: A list (or other integer-indexible sequence that supports len())
        of (imageURI, identifer) pairs.
    imgdir: Directory in which to save downloaded images.
    timeout: Seconds to wait before connect or read timeouts.
    maxthread_cnt: The maximum number of threads to use.
    """
    downloadscnt = len(downloads)
    mtdownloads = MTListReader(downloads)

    if downloadscnt >= maxthread_cnt:
        tcnt = maxthread_cnt
    else:
        tcnt = downloadscnt

    outputq = MTQueue()

    threads = []
    for cnt in range(tcnt):
        thread = ImageDownloadWorker(mtdownloads, outputq, imgdir, timeout)
        thread.daemon = True
        threads.append(thread)
        thread.start()

    lasterror = None

    resultcnt = 0
    while resultcnt < downloadscnt and not(mtdownloads.error.is_set()):
        result = outputq.get()
        resultcnt += 1
        if result.result == ImageDownloadWorkerResult.ERROR:
            mtdownloads.error.set()
            lasterror = result
        else:        
            yield result

    for thread in threads:
        thread.join()

    if lasterror is not None:
        raise lasterror.exception

