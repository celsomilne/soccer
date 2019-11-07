import cv2
from bounding_box import bounding_box as bb
import xmltodict
img = "280.jpg"
xml = "280.xml"

img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)