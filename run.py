import os
from src.view import Visualiser
from src.model import SoccerModel
from src.model import ModelBase
from src.model import SoccerObjectDetector


class NotModel(ModelBase):
    def __init__(self, detector: SoccerObjectDetector):
        super().__init__()
        self.detector = detector
        return

    def predict(self, frameNo, prevFrame):
        df = self.detector.get_df(batchNum=frameNo).reset_index()
        pred = list()
        for idx, row in df.iterrows():
            label = row["label"]
            left = int(row["left"])
            top = int(row["top"])
            right = int(left + row["width"])
            bottom = int(top + row["height"])
            pt1 = (left, top)
            pt2 = (right, bottom)
            pred.append((label, (pt1, pt2), 20, 0.6, 0.2))
        return pred


detector = SoccerObjectDetector(bboxpath=None)
videoname = os.path.abspath("sample_data/side_to_side.mov")
detector(videoname, savedir="test/")

# Create an instance of our model
model = SoccerModel(detector)

# Create a new visualiser
cmap = {"alpha": "blue", "omega": "blue", "ball": "green", "other": "black"}
vis = Visualiser(videoname, model, cmap=cmap)

# Show all markups
vis.markup_all(os.path.abspath("test.avi"), codec="MJPG", verbose=True, show=True)
