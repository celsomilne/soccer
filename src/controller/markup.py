import cv2
import numpy as np
from bounding_box import bounding_box as bb

from ..model import ModelBase


class MarkupEngine_Base(object):

    def __init__(self):
        return

class MarkupEngine(MarkupEngine_Base):

    # Font for markups
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.6

    # TODO - make a spectrum of colours for marking up
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def markup(self, frame, prevFrame):
        # Prediction should be a list of tuples
        # -> (label, (pt1, pt2), speed)
        predictions = self.model.predict(frame, prevFrame)
        for prediction in predictions:
            label, (pt1, pt2), speed = prediction

            # Get each of the corner points
            top_left, bottom_left, top_right, bottom_right = self.parsePoints(pt1, pt2)
            left = top_left[0]
            top = top_left[1]
            right = bottom_right[0]
            bottom = bottom_right[1]

            # Convert to RGB (library requirement)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw the bounding box, text and speed
            bb.add(frame, left, top, right, bottom, "%sm/s" % (speed), 'blue')

            # Convert to BGR (for opencv)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # # TODO - make the text with background to make easier to view
            # cv2.putText(frame, "%sm/s" % (speed), bottom_right, self.font, self.fontScale, [0, 0, 0])
        return frame

    @staticmethod
    def parsePoints(pt1, pt2):
        pts = np.array(list(zip(pt1, pt2))).T

        top_left = pts.min(axis=0).tolist()
        bottom_right = pts.max(axis=0).tolist()
        bottom_left = [top_left[0], bottom_right[1]]
        top_right = [bottom_right[0], top_left[1]]

        # Convert each point to a tuple
        ret = [top_left, bottom_left, top_right, bottom_right]
        ret = [tuple(pt) for pt in ret]
        return ret