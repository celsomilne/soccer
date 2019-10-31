from ..utils.models import *
from ..utils.utils import *
from ..utils.datasets import *

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches  # mark labels on images
from matplotlib.ticker import NullLocator  # get rid of axis ticks


class SoccerObjectDetector(object):
    """ Find soccer players and save bounding boxes as a pandas DataFrame"""

    def __init__(self):
        # use pretrained darknet53 weights
        fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        utils_path = os.path.join(fpath, "utils")
        
        # path to model definition file
        self.model_def = os.path.join(
            utils_path, "yolov3.cfg"
        ) 

        # path to weights file
        self.weights_path = os.path.join(
            utils_path, "yolov3.weights"
        ) 

        # path to class label file 
        self.class_path = os.path.join(
            utils_path, "coco.names"
        )

        # path to bounding boxes  
        self.bbox_path = os.path.join(
            utils_path, "bboxes.pkl"
        )  
        self.conf_thres = 0.6  # object confidence threshold
        self.nms_thres = 0.4  # iou thresshold for non-maximum suppression
        self.img_size = 704  # this needs to be a multiple of 2^5

        self.dataframe_labels = [
            "batchNum",
            "fileName",
            "label",
            "left",
            "top",
            "width",
            "height",
        ]

        # load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)

        # load weights
        if self.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(self.weights_path)
        else:
            self.model.load_state_dict(
                torch.load(self.weights_path)
            )  # checkpoint weights

        self.model.eval()  # for inference

        self.classes = load_classes(self.class_path)  # Extracts class labels from file

    def detect_team(self, img_path: str, box: tuple = (0, 0, 1280, 704)) -> str:
        """ find the detected player's team """
        imgObj = Image.open(img_path[0])
        cropped = np.array(imgObj.resize(size=(32, 32), box=box))
        medianRed = np.median(cropped[:, :, 0])
        medianBlue = np.median(cropped[:, :, 2])

        return "alpha" if medianRed < 100 and medianBlue < 90 else "omega"

    def make_square(
        self, im: np.array, min_size: int = 704, fill_color: tuple = (0, 0, 0, 0)
    ):
        """ add zero padding and resize to required size and make square """
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new("RGB", (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        new_im = new_im.resize(size=(min_size, min_size))
        return new_im

    def __call__(self, image_folder: str) -> pd.DataFrame:
        """ load all the images and find bounding boxes """

        dataloader = DataLoader(
            ImageFolder(image_folder, img_size=self.img_size),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

        self.bb_df = pd.DataFrame(columns=self.dataframe_labels)
        Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )

        pbar = tqdm(range(len(dataloader)))  # progress bar
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            pbar.update(1)

            img = Variable(input_imgs.type(Tensor))

            with torch.no_grad():
                detections = self.model(img)
                detections = non_max_suppression(
                    detections, self.conf_thres, self.nms_thres
                )

            if detections is not None:
                detections = rescale_boxes(detections[0], self.img_size, (720, 1280))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # convert to soccer labels
                    if int(cls_pred) == 32:
                        label = "ball"
                    elif int(cls_pred) == 0:
                        box = np.int32([x1, y1, x2, y2])
                        if np.sum(box < 0) or x2 > 1280 or y2 > 720:
                            label = "other"
                            break
                        label = self.detect_team(img_paths, box=box)
                    else:
                        label = "other"

                    # append found bboxes to DataFrame
                    df = pd.DataFrame(
                        data=[
                            [
                                batch_i,
                                img_paths,
                                label,
                                *np.float32([x1, y1, x2 - x1, y2 - y1]),
                            ]
                        ],
                        columns=self.dataframe_labels,
                    )
                    self.bb_df = self.bb_df.append(df, ignore_index=True)

    def save(self):
        """ save the found bboxes """
        with open(self.bbox_path, "wb") as handle:
            pickle.dump(self.bb_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        """ load the saved bboxes """
        with open(self.bbox_path, "rb") as handle:
            # We need to modify this
            self.bb_df = pickle.load(handle)

    def get_df(self, batchNum: int):
        """ grab the bboxes for the batchNum'th image """
        instances = self.bb_df["batchNum"] == batchNum
        return self.bb_df[instances]

    def show_rand(self, batchNum: int = -1):
        """ visualise the bboxes on top of the image """
        batchNums = self.bb_df["batchNum"]
        randIdx = np.random.randint(np.max(batchNums)) if batchNum < 0 else batchNum
        instances = self.bb_df["batchNum"] == randIdx
        df = self.bb_df[instances]
        if len(df) > 0:
            fileName = df["fileName"].iloc[0][0]
            fig, ax = plt.subplots(1)
            ax.imshow(Image.open(fileName))

            for i in range(len(df)):
                bbox = patches.Rectangle(
                    (df.iloc[i][3], df.iloc[i][4]),
                    df.iloc[i][5],
                    df.iloc[i][6],
                    linewidth=2,
                    edgecolor=[0.5, 0.5, 0.5],
                    facecolor="none",
                )
                ax.add_patch(bbox)
                plt.text(
                    df.iloc[i][3],
                    df.iloc[i][4],
                    s=df.iloc[i][2],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": [0.5, 0.5, 0.5], "pad": 0},
                )
            plt.axis("off")
            plt.title(fileName)
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.show()
        else:
            print("no objects found")


if __name__ == "__main__":
    """ test the class object functionality here """
    findObjects = SoccerObjectDetector()
    # takes 3 s per image
    # findObjects(image_folder='../notebooks/objectDetectTest')
    # findObjects.save()
    findObjects.load()
    findObjects.show_rand()
