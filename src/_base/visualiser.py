import cv2
from tqdm.auto import tqdm


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
        if fname is not None:
            return cv2.VideoWriter(fname, fourcc, fr, size, True)
        return None

    def _cv_write_frame(self, writer, frameRGB):
        frameBGR = self._cvt_RGB2BGR(frameRGB)
        writer.write(frameBGR)

    def _set_curFrame(self, frame):
        self.prevFrame = self.curFrame
        self.curFrame = frame

    def _get_fCount(self, verbose, every=1, desc=None):
        fCount = range(0, self.frameCount, every)
        if verbose:
            fCount = tqdm(fCount, desc=desc)
        return fCount

    def _set_frameNum(self, index):
        if index >= self.frameCount:
            raise IndexError(
                "Index %s out of bounds for video length %s" % (index, self.frameCount)
            )
        self._set_CV_prop(cv2.CAP_PROP_POS_FRAMES, index)

    def get_frame(self, index):
        """Get the frame at a given index without changing the seed or the previous frame.
        
        Parameters
        ----------
        index : int     
            Frame index.
        
        Returns
        -------
        frame : np.ndarray
            The BGR frame, as a numpy array.
        """
        cur_frame_idx = self._get_CV_prop(cv2.CAP_PROP_POS_FRAMES)
        self._set_frameNum(index)
        frame = self.next_frame()
        self._set_frameNum(cur_frame_idx)
        return frame

    def next_frame(self):
        """Get the next frame in the capture.
        
        Returns
        -------
        [type]
            [description]
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
        self._set_frameNum(0)
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
