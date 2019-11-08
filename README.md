# soccer
Repository created for the purpose of AMME4710 - Computer Vision Major Assignment. University of Sydney 2019.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Running this code
### Dependencies
First, install project dependencies.

```bash
pip install -r requirements.txt
```

If there is an error installing Pytorch (which is likely) go the [PyTorch](https://pytorch.org/get-started/locally/#start-locally) to download the relevant version for your system.

### Get YOLO Weights
```bash
cd ./src/utils
wget -c https://pjreddie.com/media/files/yolov3.weights
cd ../..
```

### Clear Saved Results
Do this every time you want to run a new video. 
```bash
cd ./src
```
__Windows__
```shell
del /S *.pkl
```
__Mac / Linux__
```bash
find /path -name '*.pkl' -delete
```

### Modify `run.py`
In `run.py`, update the parameters at the top of the file on lines 10-13. 
```python
###############
# MODIFY HERE #
###############
VIDEO_FILENAME = "sample_data/side_to_side.mov" # Name of the video to markup
SAVE_VIDEO_FRAMES_DIR = "test/"                 # Directory to save frame jpgs to
SAVE_VIDEO = False                              # If true, will show in real time, otherwise will save
SAVE_MARKUP_TO = "output.avi"                   # Video to save video to (SAVE_VIDEO is True)
```

## Viewing Frames
To convert an output video to a folder of jpgs (for easier viewing) alter the file `./videoToFrames.py` and set the `VIDEONAME` value to be the name of the video created from `SAVE_MARKUP_TO` in the above section. Then:

```bash
python videoToFrames.py
```

Take a look at `output/` to view the frames!