import cv2
import os
import numpy as np
from bounding_box import bounding_box as bb

from ..model import ModelBase


class MarkupEngine_Base(object):

    data_dir = os.path.join("..", "data")

    def __init__(self):
        return


class MarkupEngine(MarkupEngine_Base):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def markup(self, frame, frameNo, prevFrame=None, cmap=None):
        """Markup a given frame
        
        Parameters
        ----------
        frame : np.ndarray
            The frame
        frameNo : int
            The frame number
        prevFrame : np.ndarray, int 
            The previous frame or frame number
        cmap : dictionary, optional
            Maps labels to colours, by default None
        
        Returns
        -------
        np.ndarray
            Numpy array of the frame in BGR
        """
        # Prediction should be a list of tuples
        # -> (label, (pt1, pt2), speed, x, y)
        predictions = self.model.predict(frameNo, 25)
        if predictions is None:
            return frame

        # Load the football pitch with the same height as the frame
        fbPitch = FootballPitch()
        fbPitch.load(height=frame.shape[0])

        # Go through each predicted position
        for prediction in predictions:
            label, (pt1, pt2), speed, x, y = prediction
            speed = "%.2fm/s" % (speed)
            color = self._getColorFromCmap(label, cmap)

            # Create the bounding box and draw onto the image
            bb = BoundingBox(pt1, pt2, speed, color=color)
            frame = bb.draw(img=frame, isRGB=False)

            # Draw the x and y position of the player
            fbPitch.draw(x, y, color)

        frame = fbPitch.merge(frame)
        return frame

    @staticmethod
    def _getColorFromCmap(label, cmap):
        color = cmap.get(label)
        return color

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


class BoundingBox(object):
    def __init__(self, pt1, pt2, label, color="blue"):
        self.pt1 = pt1
        self.pt2 = pt2
        self.label = label
        self.color = color

        # Get the boundaries of the box
        self.left, self.top, self.right, self.bottom = self._getBoundaries(pt1, pt2)

    def draw(self, img, isRGB=False):
        if not isRGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(self.left)
        # print(self.right)
        # print(self.bottom)
        # print(self.top)
        bb.add(
            img, self.left, self.top, self.right, self.bottom, self.label, self.color
        )
        if not isRGB:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _getBoundaries(self, pt1, pt2):
        # Get each of the corners
        top_left, bottom_left, top_right, bottom_right = self._getCorners(pt1, pt2)
        left = top_left[0]
        top = top_left[1]
        right = bottom_right[0]
        bottom = bottom_right[1]
        return left, top, right, bottom

    @staticmethod
    def _getCorners(pt1, pt2):
        pts = np.array(list(zip(pt1, pt2))).T

        top_left = pts.min(axis=0).tolist()
        bottom_right = pts.max(axis=0).tolist()
        bottom_left = [top_left[0], bottom_right[1]]
        top_right = [bottom_right[0], top_left[1]]

        # Convert each point to a tuple
        ret = [top_left, bottom_left, top_right, bottom_right]
        ret = [tuple(pt) for pt in ret]
        return ret


class FootballPitch(object):

    fdir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.abspath(os.path.join(fdir, "..", "data"))

    def __init__(self):
        base = "football_pitch"
        self.fname = os.path.join(self.data_dir, f"{base}.jpg")
        self.markup = os.path.join(self.data_dir, f"{base}.xml")
        self.ratio = 1

    def load(self, height=None):
        img = cv2.imread(self.fname)
        img = self.resize(img, height)
        self._img = img

    def draw(self, x, y, color=None):
        pitch = self.pitch
        xmin, ymin, xmax, ymax = self.bb
        xCoord = int(xmin + x * (xmax - xmin))
        yCoord = int(ymin + y * (ymax - ymin))
        cv2.circle(pitch, (xCoord, yCoord), 10, [255, 0, 0], thickness=cv2.FILLED)

    def merge(self, img):
        # Assumes that the image is in BGR

        # Dimensions of the image to merge with
        h, w, _ = img.shape

        # Scale down ratio
        r = 1 / 3
        ratio = self.pitch.shape[0] / h * r

        # Put the field in the top right hand side of the image
        pitch = self.resizeRatio(self.pitch, ratio)

        # Set the top right
        pitchH, pitchW, _ = pitch.shape
        maxh = 0 + pitchH
        minw = w - pitchW
        img[0:maxh, minw:w, :] = pitch
        return img

    def resize(self, img, height=None):
        # Get the image dimensions
        h, w, _ = img.shape
        if height is None:
            height = h

        # Find the scaling ratio
        ratio = height / h
        self.ratio = ratio
        return FootballPitch.resizeRatio(img, ratio)

    def _setImg(self, img):
        self._img = img

    @staticmethod
    def resizeRatio(img, ratio):
        h, w, _ = img.shape
        w *= ratio
        h *= ratio
        # Resize
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        return img

    @property
    def pitch(self):
        if self._img is None:
            self.load()
        return self._img

    @property
    def bb(self):
        # TODO - pull from XML
        ratio = self.ratio
        return 22, 22, 530, 698
