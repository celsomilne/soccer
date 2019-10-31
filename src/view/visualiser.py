import cv2
import numpy as np
from tqdm.auto import tqdm

from . import MarkupEngine
from .._base import Visualiser_Base


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
