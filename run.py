from src.view import Visualiser
from src.model import SoccerModel
from src.model import ModelBase

class NotModel(ModelBase):
    def __init__(self):
        return
    def predict(self, frame, prevFrame):
        pred = ("testLabel", ((180, 360), (1200, 50)), 20)
        return pred
    
# Create an instance of our model
model = NotModel()

# Create a new visualiser
vis = Visualiser("notebooks/training/train_1.avi", model)

# Show all markups
vis.markup_all("test.avi", codec="MPG4", verbose=True, show=True)