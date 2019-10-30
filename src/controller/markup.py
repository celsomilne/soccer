import cv2
import numpy as np

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

            # Draw the bounding box, text and speed
            cv2.rectangle(frame, pt1, pt2, [255, 0, 0])

            # TODO - make the text with background to make easier to view
            cv2.putText(frame, label, top_left, self.font, self.fontScale, [0, 0, 0])
            cv2.putText(frame, "%sm/s" % (speed), bottom_right, self.font, self.fontScale, [0, 0, 0])
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