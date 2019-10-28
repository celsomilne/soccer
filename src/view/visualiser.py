import cv2
import numpy as np
from tqdm.auto import tqdm

from . import MarkupEngine

class Visualiser_Base(object):

    def __init__(self, video_io):
        self.capture = self._read_io(video_io)

        # Video properties
        self.frameCount = int(self._get_CV_prop(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(self._get_CV_prop(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self._get_CV_prop(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameRate = int(self._get_CV_prop(cv2.CAP_PROP_FPS))

    def _read_next(self):
        return self.capture.read()

    def _get_CV_prop(self, prop):
        return self.capture.get(prop)

    def _set_CV_prop(self, prop, *args, **kwargs):
        self.capture.set(prop, *args, **kwargs)

    def _create_new_writer(self, fname, fourcc):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        size = (self.frameWidth, self.frameHeight)
        fr = self.frameRate
        return cv2.VideoWriter(fname, fourcc, fr, size, True)

    def _cv_write_frame(self, writer, frameRGB):
        frameBGR = self._cvt_RGB2BGR(frameRGB)
        writer.write(frameBGR)

    def _reset_seed(self):
        self._set_CV_prop(cv2.CAP_PROP_POS_FRAMES, 0)
        self.prevFrame = None

    def _set_curFrame(self, frame):
        self.prevFrame = self.curFrame
        self.curFrame = frame

    def release(self):
        self.frameCount = self.frameHeight = self.frameWidth = self.frameRate = 0
        self.capture.release()

    @staticmethod
    def _cvt_BGR2RGB(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _cvt_RGB2BGR(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def _read_io(video_io):
        _io = None
        if isinstance(video_io, cv2.VideoCapture):
            _io = video_io
        elif isinstance(video_io, str):
            _io = cv2.VideoCapture(video_io)
        return _io
        

class Visualiser(Visualiser_Base):

    def __init__(self, video_io, model):
        super().__init__(video_io)
        self.MUE = MarkupEngine(model)

        # Keep track of previous and current frame
        self.prevFrame = None
        self.curFrame = None
    
    def next_frame(self):
        """
        Get the next frame in the capture.
        """
        hasFrame, frame = self._read_next()
        self._set_curFrame(frame)
        if hasFrame:
            frame = self._cvt_BGR2RGB(frame)
        return hasFrame, frame

    def next_markup(self):
        """
        Get the next marked up frame
        """
        hasFrame, frame = self.next_frame()
        if hasFrame:
            frame = self.MUE.markup(frame, prevFrame=self.prevFrame)
        return hasFrame, frame

    def markup_all(self, fname, codec="", verbose=True):
        vw = self._create_new_writer(fname, codec)
        
        # Range of frames to iterate over
        fCount = range(self.frameCount)
        if verbose:
            fCount = tqdm(fCount)

        # Iterate through each frame
        for fi in fCount:
            hasFrame, frame = self.next_markup()

            # Break early if we have no frame
            if not hasFrame:
                break

            # Write to the video writer
            self._cv_write_frame(vw, frame)

        # Reset the position
        self._reset_seed()
