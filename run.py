import os
from src.view import Visualiser
from src.model import SoccerModel
from src.model import ModelBase
from src.model import SoccerObjectDetector

###############
# MODIFY HERE #
###############
VIDEO_FILENAME = "sample_data/side_to_side.mov" # Name of the video to markup
SAVE_VIDEO_FRAMES_DIR = "side_to_side/"         # Directory to save frame jpgs to
SAVE_VIDEO = False                              # If true, will show in real time, otherwise will save
SAVE_MARKUP_TO = "output.avi"                   # Video to save video to (SAVE_VIDEO is True)

#####################################
########### DO NOT MODIFY ###########
#####################################

detector = SoccerObjectDetector(bboxpath=None)
videoname = os.path.abspath(VIDEO_FILENAME)
savedir = os.path.abspath(SAVE_VIDEO_FRAMES_DIR)
detector(videoname, savedir=savedir)

# Create an instance of our model
model = SoccerModel(detector)

# Create a new visualiser
cmap = {"alpha": "blue", "omega": "blue", "ball": "green", "other": "black"}
vis = Visualiser(videoname, model, cmap=cmap)

# Show all markups
saveto = os.path.abspath(SAVE_MARKUP_TO)
show = not SAVE_VIDEO
vis.markup_all(saveto, codec="MJPG", verbose=True, show=show)
