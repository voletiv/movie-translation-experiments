import csv
import cv2
import dlib
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import tqdm

from skimage.transform import resize

# Variables
DATASET_DIR = '/home/voletiv/Datasets/MOVIE_TRANSLATION/'
YOUTUBE_VIDEOS_DIR ='/home/voletiv/Datasets/MOVIE_TRANSLATION/in_progress/'
SHAPE_PREDICTOR_PATH = '/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat'


# videos
#     english
#         LDC
#             LDC_0000.mp4
#             LDC_0001.mp4
#         GC
#             GC_0000.mp4
#             GC_0001.mp4

#     hindi
#         SK
#         SRK

#     telugu
#         Mahesh_Babu
#         NTR
#         Sharwanand

# metadata
#     english
#         LDC.txt
#         GC.txt

#     hindi
#         SK.txt
#         SRK.txt

#     telugu
#         Mahesh_Babu.txt
#         NTR.txt
#         Sharwanand.txt

# frames
#     english
#         LDC
#             LDC_0000
#                 LDC_0000.gif
#                 LDC_0000_frame_000.png
#                 LDC_0000_frame_001.png
#             LDC_0001
#         GC

#     hindi
#         SK
#         SRK

#     telugu
#         Mahesh_Babu
#         NTR
#         Sharwanand

# landmarks
#     english
#         LDC
#             LDC_0000_landmarks.txt
#             LDC_0000_landmarks.txt
#         GC

#     hindi
#         SK
#         SRK

#     telugu
#         Mahesh_Babu
#         NTR
#         Sharwanand

