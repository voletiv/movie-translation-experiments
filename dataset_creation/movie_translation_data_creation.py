from movie_translation_data_creation_params import *
from movie_translation_data_creation_functions import *

config = MovieTranslationConfig()

# Load detector, predictor
detector, predictor = load_detector_and_predictor()


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
        metadata = read_metadata(metadata_txt_file)
        # Extract video clips
        print("Extracting video clips...")
        extract_video_clips(language, actor, metadata, verbose=False)
        video_clips_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'videos', language, actor)
        # Extract faces and landmarks from video clips
        print("Extracting faces and landmarks from video clips...")
        for video_file in tqdm.tqdm(sorted(glob.glob(os.path.join(video_clips_dir, "*.mp4")))):
            extract_face_frames_from_video(video_file, detector, predictor, save_with_blackened_mouths_and_polygons=True, save_gif=False, save_landmarks_as_txt=True)


# ONLY IF NOT DONE DURING PREVIOUS STEP!!!
# # Make mouth_blacked_and_keypoints_polygon images
# for language in tqdm.tqdm(sorted(os.listdir(os.path.join(MOVIE_TRANSLATION_DATASET_DIR, 'frames')))):
#     for actor in tqdm.tqdm(sorted(os.listdir(os.path.join(MOVIE_TRANSLATION_DATASET_DIR, 'frames', language)))):
#         for video_name in tqdm.tqdm(sorted(os.listdir(os.path.join(MOVIE_TRANSLATION_DATASET_DIR, 'frames', language, actor)))):
#             # Make mouth_blacked_and_keypoints_polygon images for each video
#             make_blackened_mouths_and_mouth_polygons(video_name)

######################################
# Split all into train, val and test,
 # and save in output_dir
######################################

output_dir = '/home/voletiv/GitHubRepos/DeepLearningImplementations/pix2pix/data/Mahesh_Babu_black_mouth_polygons'
language = 'telugu'
actor = 'Mahesh_Babu'

output_dir_train = os.path.join(output_dir, 'train')
output_dir_val = os.path.join(output_dir, 'val')
output_dir_test = os.path.join(output_dir, 'test')

if not os.path.exists(output_dir_train):
    os.makedirs(output_dir_train)

if not os.path.exists(output_dir_val):
    os.makedirs(output_dir_val)

if not os.path.exists(output_dir_test):
    os.makedirs(output_dir_test)

# Read all frame names
all_frames = []
for video_name in tqdm.tqdm(sorted(glob.glob(os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'frames_combined', language, actor, '*/')))):
    for frame in sorted(glob.glob(os.path.join(video_name, '*'))):
        all_frames.append(frame)

# Shuffle the frame names
np.random.seed(29)
np.random.shuffle(all_frames)

train_set_len = int(len(all_frames) * 0.8)
val_set_len = int(len(all_frames) * 0.1)
test_set_len = int(len(all_frames) * 0.1)

for frame in tqdm.tqdm(all_frames[:train_set_len]):
    a = subprocess.call(['cp', frame, output_dir_train])

for frame in tqdm.tqdm(all_frames[train_set_len:(train_set_len + val_set_len)]):
    a = subprocess.call(['cp', frame, output_dir_val])

for frame in tqdm.tqdm(all_frames[(train_set_len + val_set_len):(train_set_len + val_set_len + test_set_len)]):
    a = subprocess.call(['cp', frame, output_dir_test])




