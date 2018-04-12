import glob
import numpy
import os
import subprocess
import tqdm

from make_andrew_ng_dataset_functions_every_nth_frame import *

ANDREW_NG_DIR = '/shared/fusor/home/voleti.vikram/ANDREW_NG/'

#####################################################
# Extract Andrew_Ng's face frames
####################################################

# for vid_dir in glob.glob(os.path.join(ANDREW_NG_DIR, 'videos', '*/')):

for vid in sorted(glob.glob(os.path.join(ANDREW_NG_DIR, 'videos', 'CV', '*.mp4'))):
    extract_person_face_frames(vid,
                               out_dir=ANDREW_NG_DIR,
                               person_name='andrew_ng', person_face_image='/shared/fusor/home/voleti.vikram/ANDREW_NG/andrew_ng.png',
                               shape_predictor_path='/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat',
                               face_rec_model_path='/shared/fusor/home/voleti.vikram/dlib_face_recognition_resnet_model_v1.dat',
                               overwrite_frames=True, overwrite_face_shapes=False, save_faces=False)

# Frames have been manually filtered for goodness and placed in:
# /shared/fusor/home/voleti.vikram/ANDREW_NG_CLEAN/faces_combined
ANDREW_NG_CLEAN_DIR = '/shared/fusor/home/voleti.vikram/ANDREW_NG_CLEAN/'

#####################################################
# Shuffle and place in train, val, test
#####################################################

DATA_DIR = '/shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/data/andrew_ng/'

data_dir_train = os.path.join(DATA_DIR, 'train/')
data_dir_val = os.path.join(DATA_DIR, 'val/')
data_dir_test = os.path.join(DATA_DIR, 'test/')

if not os.path.exists(data_dir_train):
    os.makedirs(data_dir_train)

if not os.path.exists(data_dir_val):
    os.makedirs(data_dir_val)

if not os.path.exists(data_dir_test):
    os.makedirs(data_dir_test)

# Read all frame names
all_frame_names = []
for frame_name in tqdm.tqdm(sorted(glob.glob(os.path.join(ANDREW_NG_CLEAN_DIR, 'faces_combined', '*/', 'andrew_ng', '*')))):
    all_frame_names.append(frame_name)

# Shuffle the frame names
np.random.seed(29)
np.random.shuffle(all_frame_names)

# Set train, val, test video names
train_set_len = round(len(all_frame_names) * 0.9)
val_set_len = round(len(all_frame_names) * 0.05)
test_set_len = round(len(all_frame_names) * 0.05)

# Copy train frames
for frame_name in tqdm.tqdm(all_frame_names[:train_set_len]):
    a = subprocess.call(['cp', frame_name, data_dir_train])

# Copy val frames
for frame_name in tqdm.tqdm(all_frame_names[train_set_len:(train_set_len + val_set_len)]):
    a = subprocess.call(['cp', frame_name, data_dir_val])

# Copy test frames
for frame_name in tqdm.tqdm(all_frame_names[(train_set_len + val_set_len):(train_set_len + val_set_len + test_set_len)]):
    a = subprocess.call(['cp', frame_name, data_dir_test])

