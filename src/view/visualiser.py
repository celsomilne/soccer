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

        # Keep track of previous and current frame
        self.prevFrame = None
        self.curFrame = None

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

    def _set_curFrame(self, frame):
        self.prevFrame = self.curFrame
        self.curFrame = frame

    def _get_fCount(self, verbose, every=1):
        fCount = range(0, self.frameCount, every)
        if verbose:
            fCount = tqdm(fCount)
        return fCount

    def next_frame(self):
        """
        Get the next frame in the capture.
        """
        hasFrame, frame = self._read_next()
        self._set_curFrame(frame)
        if hasFrame:
            frame = self._cvt_BGR2RGB(frame)
        return hasFrame, frame

    def release(self):
        self.frameCount = self.frameHeight = self.frameWidth = self.frameRate = 0
        self.capture.release()

    def reset_seed(self):
        self._set_CV_prop(cv2.CAP_PROP_POS_FRAMES, 0)
        self.prevFrame = None

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

    @property
    def msPerFrame(self):
        return 1000 // self.frameRate


class Visualiser(Visualiser_Base):
    def __init__(self, video_io, model, cmap=dict()):
        """Create a new visualiser object.
        
        Parameters
        ----------
        video_io : str or cv2.VideoCapture 
            If string, will create a new cv2.VideoCapture object from filename.
        model : src.Model._base.ModelBase
            A model class. Must implement the method `predict(self, frame, prevFrame) 
        cmap : dict
            Dictionary mapping labels to particular labels. Keys should be labels, values should be colours. Possible values are: navy, blue, aqua, teal, olive, green, lime, yellow, orange, red, maroon, fuchsia, purple, black, gray , silver. If None is given, a random colour will be chosen.
        """
        super().__init__(video_io)
        self.MUE = MarkupEngine(model)
        self.cmap = cmap

    def next_markup(self):
        """
        Get the next marked up frame
        """
        hasFrame, frame = self.next_frame()
        if hasFrame:
            frame = self.MUE.markup(frame, self.prevFrame, cmap=self.cmap)
        return hasFrame, frame

    def markup_all(self, fname, codec="MPG4", verbose=True, show=False):
        if not show:
            vw = self._create_new_writer(fname, codec)
        else:
            window = cv2.namedWindow(fname)
        self.reset_seed()

        # Range of frames to iterate over
        fCount = self._get_fCount(verbose)

        # Iterate through each frame
        for fi in fCount:
            hasFrame, frame = self.next_markup()

            # Break early if we have no frame
            if not hasFrame:
                break

            if show:
                frame = self._cvt_BGR2RGB(frame)
                # Show the frame
                cv2.imshow(fname, frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(self.msPerFrame) & 0xFF == ord("q"):
                    break
            else:
                # Write to the video writer
                self._cv_write_frame(vw, frame)

        # Reset the position
        self.reset_seed()
