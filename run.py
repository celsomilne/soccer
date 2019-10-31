import os
from src.view import Visualiser
from src.model import SoccerModel
from src.model import ModelBase
from src.model import SoccerObjectDetector

class NotModel(ModelBase):
    def __init__(self):
        super().__init__()
        return
    def predict(self, frame, prevFrame):
        pred = [("testLabel", ((180, 360), (1200, 50)), 20, 0.6, 0.2)]
        return pred
    
detector = SoccerObjectDetector()
videoname = os.path.abspath("sample_data/middle_field_1.mov")
detector(videoname, savedir="test/")
print(detector.bb_df)

# Create an instance of our model
model = SoccerModel(detector)

# Create a new visualiser
vis = Visualiser("notebooks/training/train_1.avi", model)

# Show all markups
vis.markup_all("test.avi", codec="MPG4", verbose=True, show=True)