import dlib
import imageio
import tqdm

import utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

frames = imageio.get_reader(video_file_name)

landmarks_in_frames = []

for frame in tqdm.tqdm(frames):
    faces = detector(frame, 1)
    if len(faces) > 0:
        face = faces[0]
        shape = predictor(frame, face)
        landmarks = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
    landmarks_in_frames.append(landmarks)

lm = np.round(landmarks).astype('int')


def extract_landmarks(video_file_name, detector=None, predictor=None):
    frames = imageio.get_reader(video_file_name)
    if detector is None or predictor is None:
        detector, predictor = utils.load_dlib_detector_and_predictor()
    landmarks_in_frames = []
    for frame in tqdm.tqdm(frames):
        landmarks_in_this_frame = utils.get_landmarks_using_dlib_detector_and_predictor(frame, detector, predictor)
        if landmarks_in_this_frame is not None:
            landmarks_in_frames.append(landmarks_in_this_frame)
        else:
            landmarks_in_frames.append(landmarks_in_frames[-1])
    return np.array(landmarks_in_frames)

