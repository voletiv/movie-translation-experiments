import dlib
import glob
import imageio
import joblib
import numpy as np
import os
import time

from tqdm import tqdm

import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import utils
from config import *

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

config = MovieTranslationConfig()


def extract_person_face_frames(video_file, out_dir, person_name,
                               person_face_descriptor=None, person_face_image=None,
                               dlib_face_detector=None, dlib_shape_predictor=None, dlib_facerec=None,
                               cnn_face_detector_path=None, cnn_batch_size=128,
                               shape_predictor_path=None, face_rec_model_path=None,
                               check_for_face_every_nth_frame=10,
                               resize_to_shape=(256, 256),
                               overwrite_frames=False, overwrite_face_shapes=False,
                               save_faces=False,
                               save_faces_combined_with_blackened_mouths_and_lip_polygons=True,
                               skip_frames=0,
                               fast=True, prev_mean=np.array([726., 241.]),
                               verbose=False):
    """!@brief Reads frames, checks for person in every 10th frame; if person is present,
    processes all 10 frames for landmarks - detects faces in each frame, chooses the right
    face using dlib's recognition model, saves the frame, face, and landmarks in the right
    dirs in out_dir, (optional) combines face with black mouth polygon and saves that in
    the right dir in out_dir
    
    The right face is chosen using either "person_face_image" or
    "person_face_descriptor" as reference, and using the shape predictor
    "dlib_predictor" or loading it from "shape_predictor_path", and using
    the face recognition model "dlib_facerec" or loading it from
    "face_rec_model_path"

    E.g.:
    -----
    extract_person_face_frames('/shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV/02.C4W1L02 Edge Detection Examples.mp4',
                               '/shared/fusor/home/voleti.vikram/ANDREW_NG/pilot/old_dataset/videos/b',
                               person_name='andrew_ng',
                               person_face_image='/shared/fusor/home/voleti.vikram/ANDREW_NG/andrew_ng.png',
                               shape_predictor_path='/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat',
                               face_rec_model_path='/shared/fusor/home/voleti.vikram/dlib_face_recognition_resnet_model_v1.dat',
                               overwrite_frames=True, overwrite_face_shapes=True, save_faces=True, verbose=True)
    
    @param video_file Name of the video file whose frames and faces are to be
    extracted, for e.g. '/shared/fusor/home/voleti.vikram/ANDREW_NG/videos/1.mp4'

    @param out_dir Directory in which to save stuff (see Directory structure below)

    @param person_name Name of the person whose face we're choosing, e.g. Andrew_Ng

    @param person_face_descriptor 128-dimensional face descriptor of <person_name>
    extracted using dlib's face recognition model; provide this or <person_face_image>

    @param person_face_image Reference image of <person_name> to extract; provide this
    or <person_face_descriptor>

    @param dlib_face_detector dlib.get_frontal_face_detector() or
    dlib.cnn_face_detection_model_v1(<cnn_face_detector_path>); provide this or
    <cnn_face_detector_path>, providing this is useful in case of running this function on
    multiple <video_file>s in a loop; <cnn_face_detector_path> should point to
    mmod_human_face_detector.dat, which can be downloaded from
    http://dlib.net/files/mmod_human_face_detector.dat.bz2

    @param dlib_shape_predictor dlib.shape_predictor(shape_predictor_path); provide this or
    <shape_predictor_path>, providing this is useful in case of running this function on
    multiple <video_file>s in a loop; <shape_predictor_path> should point to
    shape_predictor_68_face_landmarks.dat or shape_predictor_5_face_landmarks.dat, which
    can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    or http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2; for more details see
    http://dlib.net/face_landmark_detection.py.html

    @param dlib_facerec dlib.face_recognition_model_v1(face_rec_model_path); provide this
    or <face_rec_model_path>, providing this is useful in case of running this function on
    multiple <video_file>s in a loop; <face_rec_model_path> should point to
    dlib_face_recognition_resnet_model_v1.dat, which can be downloaded from
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2; for more details,
    see http://dlib.net/face_recognition.py.html

    #param cnn_face_detector_path Path to mmod_human_face_detector.dat, which can be downloaded
    from http://dlib.net/files/mmod_human_face_detector.dat.bz2; for more details, see
    http://dlib.net/cnn_face_detector.py.html

    @param shape_predictor_path Path to shape_predictor_68_face_landmarks.dat or
    shape_predictor_5_face_landmarks.dat, which can be downloaded from
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 or
    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2; for more details see
    http://dlib.net/face_landmark_detection.py.html

    @param face_rec_model_path Path to dlib_face_recognition_resnet_model_v1.dat, which
    can be downloaded from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat,bz2;
    for more details see http://dlib.net/face_landmark_detection.py.html

    @param resize_to_shape Shape to resize the face frame to, e.g. (256, 256); if None,
    original face size (1.5x the size of face bounding box extracted using dlib landmarks)
    is maintained

    @param overwrite_frames To read video, extract frames, and save those frames even if
    already present; else, read the frames saved in the right dir in out_dir

    @param overwrite_face_shapes To extract face shapes using dlib's face recognition
    model even if already extracted, or not

    @param save_faces To save extracted, squared, expanded faces, or not

    @param save_faces_combined_with_blackened_mouths_and_lip_polygons Pretty obvious
    """

    # Directory structure:
    # -------------------
    # out_dir/
    #    |
    #    - videos/
    #    |    |
    #    |    - <video_file_name>.mp4
    #    | 
    #    - frames/
    #    |    |
    #    |    -<video_file_name>/
    #    |    |    |
    #    |    |    - <video_file_name>_frame_00000.png
    #    |    |    - <video_file_name>_frame_00001.png
    #    |
    #    - ref_faces/
    #    |    |
    #    |    <person_name>.png
    #    |
    #    - ref_descriptors/
    #    |    |
    #    |    <person_name>.npy
    #    |
    #    - raw_faces/
    #    |    |
    #    |    - <video_file_name>/
    #    |    |    |
    #    |    |    -<person_name>/
    #    |    |    |    |
    #    |    |    |    - <video_file_name>_frame_00000_raw_face_<person_name>.png
    #    |    |    |    - <video_file_name>_frame_00002_raw_face_<person_name>.png
    #    |
    #    - faces/
    #    |    |
    #    |    - <video_file_name>/
    #    |    |    |
    #    |    |    -<person_name>/
    #    |    |    |    |
    #    |    |    |    - <video_file_name>_frame_00000_face_<person_name>.png
    #    |    |    |    - <video_file_name>_frame_00002_face_<person_name>.png
    #    |
    #    - faces_combined/
    #    |    |
    #    |    - <video_file_name>/
    #    |    |    |
    #    |    |    - <person_name>/
    #    |    |    |    |
    #    |    |    |    - <video_file_name>_frame_00000_face_combined_<person_name>.png
    #    |    |    |    - <video_file_name>_frame_00002_face_combined_<person_name>.png
    #    |
    #    - face_shapes_in_frames_all_persons/
    #    |    |
    #    |    - <video_file_name>_landmarks_in_frames_all_persons.txt
    #    |
    #    - landmarks_in_frames_person/
    #    |    |
    #    |    - <video_file_name>_landmarks_in_frames_<person_name_1>.txt
    #    |    - <video_file_name>_landmarks_in_frames_<person_name_2>.txt
    #    |
    #    - landmarks_in_faces_person/
    #    |    |
    #    |    - <video_file_name>_landmarks_in_face_frames_<person_name_1>.txt
    #    |    - <video_file_name>_landmarks_in_face_frames_<person_name_2>.txt

    # dlib face detector
    if dlib_face_detector is None:
        if cnn_face_detector_path is None:
            dlib_face_detector = dlib.get_frontal_face_detector()
        else:
            dlib_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)

    # dlib facial landmarks detector (shape predictor)
    if dlib_shape_predictor is None:
        print("dlib shape predictor is not given. Loading from shape_predictor_path")
        if shape_predictor_path is None or not os.path.exists(shape_predictor_path):
            print("ERROR: shape_predictor_path does not exist! Given:", shape_predictor_path)
            return
        else:
            dlib_shape_predictor = dlib.shape_predictor(shape_predictor_path)

    # dlib face recognizer
    if dlib_facerec is None:
        print("dlib facerec model is not given. Loading from face_rec_model_path")
        if face_rec_model_path is None or not os.path.exists(face_rec_model_path):
                print("ERROR: face_rec_model_path does not exist! Given:", face_rec_model_path)
                return
        else:
            dlib_facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    # person_face_descriptor
    if person_face_descriptor is None:
        print("person_face_descriptor not given, trying to extract using image person_face_image:", person_face_image)
        # If person_face_image does not exist, return
        if not os.path.exists(person_face_image):
            print("ERROR: person_face_image does not exist! Given:", person_face_image)
            return
        # Else, extract person_face_descriptor from person_face_image
        ref_frame = imageio.imread(person_face_image)
        ref_face = dlib_face_detector(ref_frame, 1)[0]
        ref_shape = dlib_shape_predictor(ref_frame, ref_face)
        person_face_descriptor = np.array(dlib_facerec.compute_face_descriptor(ref_frame, ref_shape))

    # video_file_name
    video_file_split = video_file.split('/')
    videos_index = video_file_split.index('videos')
    video_file_name = os.path.splitext('_'.join(video_file_split[videos_index+1:]))[0].replace(" ", "_")
    print("VIDEO_FILE_NAME:", video_file_name)

    # Frames dir
    frames_dir = os.path.join(out_dir, 'frames', video_file_name)
    if not os.path.exists(frames_dir):
        print("Making dir", frames_dir)
        os.makedirs(frames_dir)

    # Face frames dir
    faces_dir = os.path.join(out_dir, 'faces', video_file_name, person_name)
    if save_faces and not os.path.exists(faces_dir):
        print("Making dir", faces_dir)
        os.makedirs(faces_dir)

    # Combined faces dir : for pic of face + face_with_black_mouth_polygon
    faces_combined_dir = os.path.join(out_dir, 'faces_combined', video_file_name, person_name)
    if save_faces_combined_with_blackened_mouths_and_lip_polygons and not os.path.exists(faces_combined_dir):
        print("Making dir", faces_combined_dir)
        os.makedirs(faces_combined_dir)

    # All persons' face shapes in frames dir
    face_shapes_in_frames_all_persons_dir = os.path.join(out_dir, 'face_shapes_in_frames_all_persons')
    if not os.path.exists(face_shapes_in_frames_all_persons_dir):
        print("Making dir", face_shapes_in_frames_all_persons_dir)
        os.makedirs(face_shapes_in_frames_all_persons_dir)

    # Person's landmarks in frames
    landmarks_in_frames_person_dir = os.path.join(out_dir, 'landmarks_in_frames_person')
    if not os.path.exists(landmarks_in_frames_person_dir):
        print("Making dir", landmarks_in_frames_person_dir)
        os.makedirs(landmarks_in_frames_person_dir)

    # Person's landmarks in face frames
    landmarks_in_faces_person_dir = os.path.join(out_dir, 'landmarks_in_faces_person')
    if not os.path.exists(landmarks_in_faces_person_dir):
        print("Making dir", landmarks_in_faces_person_dir)
        os.makedirs(landmarks_in_faces_person_dir)

    # To save shapes and landmarks
    face_shapes_in_frames_all_persons = []
    person_frames_landmarks = []
    person_faces_landmarks = []
    face_shapes_in_frames_all_persons_pkl_file_name = os.path.join(face_shapes_in_frames_all_persons_dir, video_file_name + '_face_shapes_in_frames_all_persons.pkl')
    landmarks_in_frames_person_txt_file_name = os.path.join(landmarks_in_frames_person_dir, video_file_name + '_landmarks_in_frames_' + person_name + '.txt')
    landmarks_in_faces_person_txt_file_name = os.path.join(landmarks_in_faces_person_dir, video_file_name + '_landmarks_in_faces_' + person_name + '.txt')

    # READ VIDEO

    # Read video - if need to overwrite, or if no sample frame is present
    sample_frame = os.path.join(frames_dir, video_file_name, video_file_name + '_frame_00000.png')
    if overwrite_frames or not os.path.exists(sample_frame):
        print("Reading video", video_file)
        video_frames_reader = imageio.get_reader(video_file)
        save_frames = True
    else:
        print("Reading frames from", frames_dir)
        video_frames_reader = glob.glob(os.path.join(frames_dir, video_file_name, '*'))
        save_frames = False

    # DETECT FACES

    # def convert_to_rect(cnn_output):
    #     return cnn_output.rect

    # # Bulk detect faces in frames
    # for i in range(0, len(video_frames), batch_size):
    #     batch_frames = video_frames[i:i+batch_size]
    #     batch_faces = cnn_face_detector(batch_frames, 1, batch_size=len(images))
    #     batch_face_rects = list(map(convert_to_rect, batch_faces))

    try:

        # Check for face only every 10th frame
        # Meanwhile, save the 10 frames of a batch in an array
        # If 10th frame has face, or prev_batch had face, check for facial landmarks, etc.
        # in all frames of batch
        batch_frame_numbers = []
        batch_frames = []
        detect_face_shapes = False

        # For each frame
        for frame_number, frame in enumerate(tqdm(video_frames_reader)):

            if frame_number < skip_frames:
                continue

            # Read frame if not from video
            if not save_frames:
                frame = imageio.imread(frame)

            # Till frame_number+1 % 10 == 0, only save the frames
            if (frame_number + 1) % check_for_face_every_nth_frame != 0:
                batch_frame_numbers.append(frame_number)
                batch_frames.append(frame)
                continue

            # At the 10th frame
            batch_frame_numbers.append(frame_number)
            batch_frames.append(frame)

            # Check if 10th frame has faces
            face_shapes_in_frame_10 = utils.get_all_face_shapes(frame, dlib_face_detector, dlib_shape_predictor)
            person_landmarks_in_frame_10 = get_person_face_lm_from_face_shapes(frame, face_shapes_in_frame_10, person_face_descriptor, dlib_facerec, fast=False)

            # If person's face is not present, and was not present in previous batch - reset and continue
            if person_landmarks_in_frame_10 is None and not detect_face_shapes:
                if verbose:
                    print("\nNo person face detected in 10th frame of batch\n")
                detect_face_shapes = False

            # Else if person's face is not present, but was present in previous batch - detect faces for all frames in batch, and deactivate detect_face_shapes
            elif person_landmarks_in_frame_10 is None and detect_face_shapes:
                if verbose:
                    print("\nNo person face detected in 10th frame of batch, but was in prev_batch, so running\n")
                for batch_frame_number, batch_frame in zip(batch_frame_numbers, batch_frames):
                    face_shapes_in_frames_all_persons, \
                    person_frames_landmarks, \
                    person_faces_landmarks, prev_mean = detect_person_face_and_shape(batch_frame, batch_frame_number,
                                                                                     video_file_name=video_file_name, person_name=person_name,
                                                                                     save_frames=save_frames, frames_dir=frames_dir,
                                                                                     save_faces=save_faces, faces_dir=faces_dir,
                                                                                     save_faces_combined_with_blackened_mouths_and_lip_polygons=save_faces_combined_with_blackened_mouths_and_lip_polygons,
                                                                                     faces_combined_dir=faces_combined_dir,
                                                                                     dlib_face_detector=dlib_face_detector, dlib_shape_predictor=dlib_shape_predictor,
                                                                                     face_shapes_in_frames_all_persons=face_shapes_in_frames_all_persons,
                                                                                     person_face_descriptor=person_face_descriptor, dlib_facerec=dlib_facerec,
                                                                                     person_frames_landmarks=person_frames_landmarks,
                                                                                     resize_to_shape=resize_to_shape,
                                                                                     person_faces_landmarks=person_faces_landmarks,
                                                                                     fast=fast, prev_mean=prev_mean)
                detect_face_shapes = False
                

            # Else if person's face is present, detect faces for all frames in batch, and activate detect_face_shapes
            elif person_landmarks_in_frame_10 is not None:
                if verbose:
                    print("\nPerson face detected in 10th frame of batch, running\n")
                for batch_frame_number, batch_frame in zip(batch_frame_numbers, batch_frames):
                    face_shapes_in_frames_all_persons, \
                    person_frames_landmarks, \
                    person_faces_landmarks, prev_mean = detect_person_face_and_shape(batch_frame, batch_frame_number,
                                                                                     video_file_name=video_file_name, person_name=person_name,
                                                                                     save_frames=save_frames, frames_dir=frames_dir,
                                                                                     save_faces=save_faces, faces_dir=faces_dir,
                                                                                     save_faces_combined_with_blackened_mouths_and_lip_polygons=save_faces_combined_with_blackened_mouths_and_lip_polygons,
                                                                                     faces_combined_dir=faces_combined_dir,
                                                                                     dlib_face_detector=dlib_face_detector, dlib_shape_predictor=dlib_shape_predictor,
                                                                                     face_shapes_in_frames_all_persons=face_shapes_in_frames_all_persons,
                                                                                     person_face_descriptor=person_face_descriptor, dlib_facerec=dlib_facerec,
                                                                                     person_frames_landmarks=person_frames_landmarks,
                                                                                     resize_to_shape=resize_to_shape,
                                                                                     person_faces_landmarks=person_faces_landmarks,
                                                                                     fast=fast, prev_mean=prev_mean)
                detect_face_shapes = True

            # Reset batch_frame_numbers and batch_frames
            batch_frame_numbers = []
            batch_frames = []
    
    except KeyboardInterrupt:
        # Clean exit by saving face shapes and landmarks
        pass

    # Save the face shapes in frames
    print("Saving", face_shapes_in_frames_all_persons_pkl_file_name)
    joblib.dump(face_shapes_in_frames_all_persons, face_shapes_in_frames_all_persons_pkl_file_name)

    # Save person frame landmarks
    print("Saving", landmarks_in_frames_person_txt_file_name)
    utils.write_landmarks_list_as_txt(landmarks_in_frames_person_txt_file_name, person_frames_landmarks)

    # Save person face landmarks
    print("Saving", landmarks_in_faces_person_txt_file_name)
    utils.write_landmarks_list_as_txt(landmarks_in_faces_person_txt_file_name, person_faces_landmarks)


def detect_person_face_and_shape(frame, frame_number, video_file_name, person_name,
                                 save_frames, frames_dir, save_faces, faces_dir,
                                 save_faces_combined_with_blackened_mouths_and_lip_polygons, faces_combined_dir,
                                 dlib_face_detector, dlib_shape_predictor,
                                 face_shapes_in_frames_all_persons,
                                 person_face_descriptor, dlib_facerec,
                                 person_frames_landmarks, resize_to_shape,
                                 person_faces_landmarks,
                                 fast=False,
                                 prev_mean=np.array([726., 241.])):

    # Image names
    video_frame_base_name = video_file_name + "_frame_{0:05d}.png".format(frame_number)
    if save_frames:
        video_frame_name = os.path.join(frames_dir, video_frame_base_name)
    video_face_base_name = video_file_name + "_frame_{0:05d}_face_{1}.png".format(frame_number, person_name)
    if save_faces:
        video_face_name = os.path.join(faces_dir, video_face_base_name)
    video_face_combined_base_name = video_file_name + "_frame_{0:05d}_face_combined_{1}.png".format(frame_number, person_name)
    if save_faces_combined_with_blackened_mouths_and_lip_polygons:
        video_face_combined_name = os.path.join(faces_combined_dir, video_face_combined_base_name)

    # Extract all face shapes in frame
    face_shapes_in_frame = utils.get_all_face_shapes(frame, dlib_face_detector, dlib_shape_predictor)

    # Append to list of face shapes in frames
    face_shapes_in_frames_all_persons.append([video_frame_base_name] + face_shapes_in_frame)

    # Get person's face shape if face is present
    person_landmarks_in_frame = get_person_face_lm_from_face_shapes(frame, face_shapes_in_frame, person_face_descriptor, dlib_facerec, fast=fast, prev_mean=prev_mean)
    
    # If person's face is present
    if person_landmarks_in_frame is not None:

        prev_mean = np.mean(person_landmarks_in_frame, axis=0)

        # Save the frame
        if save_frames:
            imageio.imwrite(video_frame_name, frame)

        # Save the landmark coordinates w.r.t. the full frame
        person_frames_landmarks.append([video_frame_base_name] + [list(l) for l in person_landmarks_in_frame])

        # Get the face - squared, expanded, resized
        face_square_expanded_resized, \
        landmarks_in_face_square_expanded_resized, _, _ = utils.get_square_expand_resize_face_and_modify_landmarks(frame,
                                                                                                                   person_landmarks_in_frame,
                                                                                                                   resize_to_shape=resize_to_shape,
                                                                                                                   face_square_expanded_resized=True)

        # Note the landmark coordinates w.r.t. the face_frame
        person_faces_landmarks.append([video_face_base_name] + [list(l) for l in landmarks_in_face_square_expanded_resized])

        # Save the face image
        if save_faces:
            imageio.imwrite(video_face_name, face_square_expanded_resized)

        # If save_with_blackened_mouths_and_polygon, combine the two
        if save_faces_combined_with_blackened_mouths_and_lip_polygons:
            face_with_blackened_mouth_and_lip_polygons = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, landmarks_in_face_square_expanded_resized[48:68])
            face_combined = np.hstack((face_square_expanded_resized, face_with_blackened_mouth_and_lip_polygons))
            imageio.imwrite(video_face_combined_name, face_combined)

    return face_shapes_in_frames_all_persons, person_frames_landmarks, person_faces_landmarks, prev_mean


def get_person_face_lm_from_face_shapes(frame, face_shapes, person_face_descriptor, dlib_facerec, fast=False, prev_mean=np.array([726., 241.])):
    '''Get the facial landmarks of a specific person
    http://dlib.net/face_recognition.py.html
    '''

    if not fast:
        # Compute the face descriptors
        face_descriptors = []
        for shape in face_shapes:
            face_descriptors.append(np.array(dlib_facerec.compute_face_descriptor(frame, shape)))

        # Compare the face descriptors
        face_descriptors_distance = np.linalg.norm(person_face_descriptor - np.array(face_descriptors).reshape(len(face_descriptors), person_face_descriptor.shape[-1]), axis=-1)

        # Choose the face with least distance, if least distance is less than 0.6
        try:
            least_dist_face = np.argmin(face_descriptors_distance)
            # If the least distant face is > 0.6, it's someone else's face
            if least_dist_face > 0.6:
                return None
        except ValueError:
            # If there are no faces in the frame, return None
            return None
 
        # Return the landamrks of their face in frame
        return utils.shape_to_landmarks(face_shapes[least_dist_face])

    else:
        if len(face_shapes) == 0:
            return None

        elif len(face_shapes) == 1:
            return utils.shape_to_landmarks(face_shapes[0])

        else:
            # Compare distances with previous mean
            distances_to_prev_mean = [np.linalg.norm(prev_mean -  np.array([(point.x, point.y) for point in face_shape.parts()]).mean(axis=0)) for face_shape in face_shapes]
            least_dist_face = np.argmin(distances_to_prev_mean)
            return utils.shape_to_landmarks(face_shapes[least_dist_face])

