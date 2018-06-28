import numpy as np
import os
import sys
import tqdm

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils


def detect_landmarks(video_frames_npy_file):
    print("---------------")
    print("Reading", video_frames_npy_file)
    video_frames = np.load(video_frames_npy_file)
    print("Loading face_alignment object")
    face_alignment_3D_object = utils.load_face_alignment_object(d='3D', enable_cuda=True)
    print("Detecting landmarks...")
    landmarks = []
    frames_with_no_landmarks = []
    for f, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):
        landmarks_in_frame = utils.get_landmarks_using_FaceAlignment(frame, face_alignment_3D_object)
        if landmarks_in_frame is not None:
            landmarks.append(landmarks_in_frame)
            frames_with_no_landmarks.append(0)
        else:
            frames_with_no_landmarks.append(1)
            if f > 0:
                landmarks.append(landmarks[-1])
            else:
                landmarks.append(np.zeros((68, 2)))
    # Save
    print("Saving my_video_landmarks.npz")
    np.savez('/tmp/my_video_landmarks', landmarks=landmarks, frames_with_no_landmarks=frames_with_no_landmarks)
    print("---------------")


if __name__ == '__main__':
    video_frames_npy_file = sys.argv[1]
    detect_landmarks(video_frames_npy_file)

