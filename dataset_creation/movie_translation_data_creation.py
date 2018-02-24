from movie_translation_data_creation_params import *
from movie_translation_data_creation_functions import *


# Load detector, predictor
detector, predictor = load_detector_and_predictor()


# Clip videos by dialogues, extract faces and landmarks from video clips
for language in tqdm.tqdm(sorted(os.listdir(os.path.join(DATASET_DIR, 'metadata')))):
    for metadata_txt_file in tqdm.tqdm(sorted(glob.glob(os.path.join(DATASET_DIR, 'metadata', language, "*")))):
        actor = os.path.splitext(os.path.basename(metadata_txt_file))[0]
        print("\n", actor, "\n")
        metadata = read_metadata(metadata_txt_file)
        # Extract video clips
        print("Extracting video clips...")
        extract_video_clips(language, actor, metadata, verbose=False)
        video_clips_dir = os.path.join(DATASET_DIR, 'videos', language, actor)
        # Extract faces and landmarks from video clips
        print("Extracting faces and landmarks from video clips...")
        for video_file in tqdm.tqdm(sorted(glob.glob(os.path.join(video_clips_dir, "*.mp4")))):
            extract_face_frames_from_video(video_file, detector, predictor, save_gif=True, save_landmarks_as_txt=True)

