import cv2
import dlib
import imageio
import math
import numpy as np
import os
import subprocess
import tqdm

if 'voletiv' in os.getcwd():
    # voletiv
    DATASET_DIR = '/home/voletiv/Datasets/MOVIE_TRANSLATION/'

elif 'voleti.vikram' in os.getcwd():
    DATASET_DIR = '/shared/fusor/home/voleti.vikram/MOVIE_TRANSLATION/'
