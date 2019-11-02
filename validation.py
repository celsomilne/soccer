import os
import cv2
from tqdm.auto import tqdm
from src.view import Visualiser
from src.model import SoccerModel
from src.model import ModelBase
from src.model import SoccerObjectDetector

# Predict bounding boxes for all of the images
savepath = os.path.abspath("./validation_bboxes.pkl")
detector = SoccerObjectDetector(savepath)
valpath = os.path.abspath(os.path.join(".", "notebooks", "training_all"))
detector(None, image_dir=valpath)

# Load all xmls into another dataframe
