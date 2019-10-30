import os
import warnings
import cv2
import pafy
import numpy as np

from ..view import Visualiser_Base


class Scraper_Base(object):
    def __init__(self, url):
        self.url = url
        self.video = None
        if url is not None:
            self.video = pafy.new(self.url)

    def _get_stream_with_resolution(self, resolution):
        if self.video is None:
            raise TypeError("No url specified.")
        streams = self.video.streams
        stream_resolutions = np.array([s.dimensions for s in streams])
        resolution = np.array(resolution)
        idx = np.where(resolution == stream_resolutions)
        stream = streams[idx[0][0]]
        return stream

    def _format_stream_title(self, stream):
        valid_chrs = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890"
        title = "".join([c if c in valid_chrs else "_" for c in stream.title])
        filename = "{}.{}".format(title, stream.extension)


class Scraper(Scraper_Base):
    def __init__(self, url):
        super().__init__(url)

    def download(self, root=".", resolution=(1280, 720)):
        # Get the stream of the right resolution
        stream = self._get_stream_with_resolution(resolution)

        # Clean the name of the stream and create the download path
        title = self._format_stream_title(stream)
        filename = os.path.join(root, title)
        self.filename = filename

        # Check if the filename already exists
        if os.path.isfile(filename):
            warnings.warn(f"Filename {filename} already exists. Aborting.")
            return

        # Download the file
        stream.download(filepath=filename, quiet=False)

    def process(self, outPath, processFun=None, every=10, filename=None, codec="MJPG"):
        if filename is None:
            filename = self.filename

        # Create our capture
        cap = CaptureObject(filename)

        # Get the file extension
        root, file_ext = os.path.splitext(outPath)
        if not file_ext:
            file_ext = "avi"
        filename = f"{root}.{file_ext}"

        # Write the video capture to file
        cap.toFile(filename, root, processFun, codec=codec, everyFrame=every)


class CaptureObject(Visualiser_Base):
    def __init__(self, video_io):
        super().__init__(video_io)

    def toFile(
        self,
        filename,
        dirname,
        processFun=None,
        codec="MJPG",
        everyFrame=10,
        *args,
        **kwargs,
    ):
        if processFun is None:
            processFun = lambda x: (True, x)
        fCount = self._get_fCount(verbose=True, every=everyFrame)
        writer = self._create_new_writer(filename, codec)

        # Make the image directory
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # Go through each frame
        for i in fCount:
            self._set_CV_prop(cv2.CAP_PROP_POS_FRAMES, i)
            hasFrame, frame = self.next_frame()
            if not hasFrame:
                break
            isValidFrame, processedFrame = processFun(frame, *args, **kwargs)

            # Write valid frames to file
            if isValidFrame:
                frame = self._cvt_RGB2BGR(frame)
                writer.write(frame)
                cv2.imwrite(f"{dirname}/{i}.png", frame)
