import numpy as np
import tqdm

import sys
sys.path.append('../')
import utils


def detect_landmarks(video_frames_npy_file):
    video_frames = np.load(video_frames_npy_file)
    face_alignment_3D_object = utils.load_face_alignment_object(d='3D', enable_cuda=True)
    landmarks = []
    frames_with_no_landmarks = []
    for f, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):
        landmarks_in_frame = utils.get_landmarks_using_FaceAlignment(frame, face_alignment_3D_object)
        if landmarks_in_frame is not None:
            landmarks.append(landmarks_in_frame[:, :2])
            frames_with_no_landmarks.append(0)
        else:
            frames_with_no_landmarks.append(1)
            if f > 0:
                landmarks.append(landmarks[-1])
            else:
                landmarks.append(np.zeros((68, 2)))
    # Save
    np.savez('/tmp/my_video_landmarks', landmarks=landmarks, frames_with_no_landmarks=frames_with_no_landmarks)


if __name__ == '__main__':
    video_frames_npy_file = sys.argv[1]
    detect_landmarks(video_frames_npy_file)

