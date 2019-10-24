import cv2
import numpy as np

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

    @staticmethod
    def _cvt_BGR2RGB(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def _read_io(video_io):
        _io = None
        if isinstance(video_io, cv2.VideoCapture):
            _io = video_io
        elif isinstance(video_io, str):
            _io = cv2.VideoCapture(video_io)
        return _io
        

class Visualiser(Visualiser_Base):

    def __init__(self, video_io):
        super().__init__(video_io)
        self.MUE = MarkupEngine()
    
    def next_frame(self):
        """
        Get the next frame in the capture.
        """
        hasFrame, frame = self._read_next()
        if hasFrame:
            frame = self._cvt_BGR2RGB(frame)
        return hasFrame, frame

    def next_markup(self):
        hasFrame, frame = self.next_frame()
        if hasFrame:
            frame = self.MUE.markup(frame)
        return hasFrame, frame
