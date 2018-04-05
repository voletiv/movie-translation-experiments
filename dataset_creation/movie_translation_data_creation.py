from movie_translation_data_creation_params import *
from movie_translation_data_creation_functions import *

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

config = MovieTranslationConfig()

# Load landmarks detector
if config.USING_DLIB_OR_FACE_ALIGNMENT == 'dlib':
    dlib_detector, dlib_predictor = load_dlib_detector_and_predictor()
elif config.USING_DLIB_OR_FACE_ALIGNMENT == 'face_alignment':
    face_alignment_object = load_face_alignment_object(enable_cuda=config.ENABLE_CUDA)

# Clip videos by dialogue times from metadata,
# extract faces and landmarks from video clips,
# blacken mouth and draw mouth polygon
# save combined frame + frame_with_blackened_mouth_and_polygon
for language in tqdm.tqdm(sorted(os.listdir(os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'metadata')))):
    # Read all metadata files in the language,
    # containing the columns: | output_file_name.mp4 | youtubeID | start_time | duration |
    for metadata_txt_file in tqdm.tqdm(sorted(glob.glob(os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'metadata', language, "*")))):
        actor = os.path.splitext(os.path.basename(metadata_txt_file))[0]
        print("\n", actor, "\n")
        metadata = utils.read_metadata(metadata_txt_file)
        # Extract video clips
        print("Extracting video clips...")
        extract_video_clips(language, actor, metadata, verbose=False)
        video_clips_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'videos', language, actor)
        # Extract faces and landmarks from video clips
        print("Extracting faces and landmarks from video clips...")
        for v, video_file in enumerate(tqdm.tqdm(sorted(glob.glob(os.path.join(video_clips_dir, "*.mp4"))))):
            # if v < 15:
            #     continue
            if config.USING_DLIB_OR_FACE_ALIGNMENT == 'dlib':
                extract_face_frames_and_landmarks_from_video(video_file, config.USING_DLIB_OR_FACE_ALIGNMENT, dlib_detector=dlib_detector, dlib_predictor=dlib_predictor,
                                                             save_with_blackened_mouths_and_polygons=True, save_gif=False, save_landmarks_as_txt=True)
            elif config.USING_DLIB_OR_FACE_ALIGNMENT == 'face_alignment':
                extract_face_frames_and_landmarks_from_video(video_file, config.USING_DLIB_OR_FACE_ALIGNMENT, face_alignment_object=face_alignment_object,
                                                             save_with_blackened_mouths_and_polygons=True, save_gif=False, save_landmarks_as_txt=True)


# ONLY IF NOT DONE DURING PREVIOUS STEP!!!
# # Make mouth_blacked_and_keypoints_polygon images
# for language in tqdm.tqdm(sorted(os.listdir(os.path.join(MOVIE_TRANSLATION_DATASET_DIR, 'frames')))):
#     for actor in tqdm.tqdm(sorted(os.listdir(os.path.join(MOVIE_TRANSLATION_DATASET_DIR, 'frames', language)))):
#         for video_name in tqdm.tqdm(sorted(os.listdir(os.path.join(MOVIE_TRANSLATION_DATASET_DIR, 'frames', language, actor)))):
#             # Make mouth_blacked_and_keypoints_polygon images for each video
#             write_combined_frames_with_blackened_mouths_and_mouth_polygons(video_name)

######################################
# Split all into train, val and test,
 # and save in output_dir
######################################

language = 'telugu'
actor = 'Mahesh_Babu'
output_dir = os.path.join(config.PIX2PIX_CODE_DIR, 'data', actor)

output_dir_train = os.path.join(output_dir, 'train')
output_dir_val = os.path.join(output_dir, 'val')
output_dir_test = os.path.join(output_dir, 'test')

if not os.path.exists(output_dir_train):
    os.makedirs(output_dir_train)

if not os.path.exists(output_dir_val):
    os.makedirs(output_dir_val)

if not os.path.exists(output_dir_test):
    os.makedirs(output_dir_test)

# Read all video names
all_video_names = []
for video_name in tqdm.tqdm(sorted(glob.glob(os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'frames_combined', language, actor, '*/')))):
    all_video_names.append(video_name)

# Shuffle the video names
np.random.seed(29)
np.random.shuffle(all_video_names)

# Set train, val, test video names
train_set_len = round(len(all_video_names) * 0.9)
val_set_len = round(len(all_video_names) * 0.05)
test_set_len = round(len(all_video_names) * 0.05)

# Train
for video_name in tqdm.tqdm(all_video_names[:train_set_len]):
    for frame in sorted(glob.glob(os.path.join(video_name, '*'))):
        a = subprocess.call(['cp', frame, output_dir_train])

# Val
for video_name in tqdm.tqdm(all_video_names[train_set_len:(train_set_len + val_set_len)]):
    for frame in sorted(glob.glob(os.path.join(video_name, '*'))):
        a = subprocess.call(['cp', frame, output_dir_val])
        
# Test
for video_name in tqdm.tqdm(all_video_names[(train_set_len + val_set_len):(train_set_len + val_set_len + test_set_len)]):
    for frame in sorted(glob.glob(os.path.join(video_name, '*'))):
        a = subprocess.call(['cp', frame, output_dir_test])
