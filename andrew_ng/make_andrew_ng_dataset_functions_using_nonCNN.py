import dlib
import glob
import imageio
import joblib
import numpy as np
import time

from tqdm import tqdm

import sys
sys.path.append('../')
import utils
from config import *

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

config = MovieTranslationConfig()


def extract_person_face_frames(video_file, out_dir, person_name,
                               person_face_descriptor=None, person_face_image=None,
                               dlib_detector=None, dlib_predictor=None, dlib_facerec=None,
                               cnn_face_detector_path=None, shape_predictor_path=None, face_rec_model_path=None,
                               resize_to_shape=(256, 256),
                               overwrite_frames=False, overwrite_face_shapes=False,
                               save_faces=False,
                               save_faces_combined_with_blackened_mouths_and_lip_polygons=True,
                               profile_time=False):
    """!@brief Extracts all frames of video_file into the right dir in
    out_dir, detects faces in each frame, chooses the right face using a
    recognition model, saves the face and landmarks in the right dirs in
    out_dir, (optional) combines face with black mouth polygon and saves
    that in the right dir in out_dir
    
    The right face is chosen using either "person_face_image" or
    "person_face_descriptor" as reference, and using the shape predictor
    "dlib_predictor" or loading it from "shape_predictor_path", and using
    the face recognition model "dlib_facerec" or loading it from
    "face_rec_model_path"

    E.g.:
    -----
    extract_person_face_frames('/shared/fusor/home/voleti.vikram/ANDREW_NG/pilot/dataset/videos/01_small.mp4',
                               '/shared/fusor/home/voleti.vikram/ANDREW_NG/pilot/',
                               'andrew_ng', person_face_image='/shared/fusor/home/voleti.vikram/ANDREW_NG/andrew_ng.png',
                               shape_predictor_path='/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat',
                               face_rec_model_path='/shared/fusor/home/voleti.vikram/dlib_face_recognition_resnet_model_v1.dat',
                               overwrite_frames=True, overwrite_face_shapes=False, save_faces=False)
    
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

    @param dlib_predictor dlib.shape_predictor(shape_predictor_path); provide this or
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

    if profile_time:
        start_time = time.time()

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
    video_file_name = os.path.splitext('_'.join(video_file_split[videos_index+1:]))[0]
    print("VIDEO_FILE_NAME:", video_file_name)

    # Frames dir
    frames_dir = os.path.join(out_dir, 'frames', video_file_name)
    if not os.path.exists(frames_dir):
        print("Making dir", frames_dir)
        os.makedirs(frames_dir)

    # Face frames dir
    if save_faces:
        faces_dir = os.path.join(out_dir, 'faces', video_file_name, person_name)
        if not os.path.exists(faces_dir):
            print("Making dir", faces_dir)
            os.makedirs(faces_dir)

    # Combined faces dir : for pic of face + face_with_black_mouth_polygon
    if save_faces_combined_with_blackened_mouths_and_lip_polygons:
        faces_combined_dir = os.path.join(out_dir, 'faces_combined', video_file_name, person_name)
        if not os.path.exists(faces_combined_dir):
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
    person_frame_landmarks = []
    person_face_landmarks = []
    face_shapes_in_frames_all_persons_pkl_file_name = os.path.join(face_shapes_in_frames_all_persons_dir, video_file_name + '_face_shapes_in_frames_all_persons.pkl')
    landmarks_in_frames_person_txt_file_name = os.path.join(landmarks_in_frames_person_dir, video_file_name + '_landmarks_in_frames_' + person_name + '.txt')
    landmarks_in_faces_person_txt_file_name = os.path.join(landmarks_in_faces_person_dir, video_file_name + '_landmarks_in_faces_' + person_name + '.txt')

    # Read video - if need to overwrite, or if no sample frame is present
    sample_frame = os.path.join(frames_dir, video_file_name, video_file_name + '_frame_00000.png')
    if overwrite_frames or not os.path.exists(sample_frame):
        print("Reading video", video_file)
        video_frames_reader = imageio.get_reader(video_file)
        video_frames = []
        for frame in video_frames_reader:
            video_frames.append(frame)
        save_frames = True
    else:
        print("Reading frames from", frames_dir)
        video_frames = glob.glob(os.path.join(frames_dir, video_file_name, '*'))
        save_frames = False

    # To detect face shapes or not
    if overwrite_face_shapes or not os.path.exists(face_shapes_in_frames_all_persons_pkl_file_name):
        print("To detect face shapes...")
        detect_face_shapes = True
    else:
        print("Reading face shapes pkl file", face_shapes_in_frames_all_persons_pkl_file_name)
        face_shapes_in_frames_all_persons = joblib.load(face_shapes_in_frames_all_persons_pkl_file_name)
        detect_face_shapes = True

    if profile_time:
        init_time = time.time()
        loop_end_time = init_time
        save_frame_dur = []
        detect_face_shapes_dur = []
        get_person_lm_dur = []
        get_modified_face_dur = []
        save_face_dur = []
        make_bmp_and_save_dur = []
        loop_dur = []

    try:

        # For each frame
        for frame_number, frame in enumerate(tqdm(video_frames)):
    
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
    
            # If overwrite_frame == True, or if no frames have been saved yet
            if save_frames:
                imageio.imwrite(video_frame_name, frame)
   
            if profile_time:
                save_frame_time = time.time()
                save_frame_dur.append(save_frame_time - loop_end_time)
 
            # Extract all face shapes in the frame, if not done already
            if detect_face_shapes:
    
                # Extract all face shapes in frame
                face_shapes_in_frame = utils.get_all_face_shapes(frame, dlib_face_detector, dlib_shape_predictor)
    
                # Append to list of face shapes in frames
                face_shapes_in_frames_all_persons.append([video_frame_base_name] + face_shapes_in_frame)
    
            else:
                face_shapes_in_frame = face_shapes_in_frames_all_persons[f]

            if profile_time:
                detect_face_shapes_time = time.time()
                detect_face_shapes_dur.append(detect_face_shapes_time - save_frame_time)
    
            # Get person's face shape if face is present
            person_landmarks_in_frame = get_person_face_lm_from_face_shapes(frame, face_shapes_in_frame, person_face_descriptor, dlib_facerec)
    
            if profile_time:
                get_person_lm_time = time.time()
                get_person_lm_dur.append(get_person_lm_time - detect_face_shapes_time)

            # If person's face is present
            if person_landmarks_in_frame is not None:
    
                # Save the landmark coordinates w.r.t. the full frame
                person_frame_landmarks.append([video_frame_base_name] + [list(l) for l in person_landmarks_in_frame])
    
                # Get the face frame - square, expanded, resized
                face_square_expanded_resized, landmarks_in_face_square_expanded_resized = utils.get_square_expand_resize_face_and_modify_landmarks(frame, person_landmarks_in_frame,
                                                                                                                                                   resize_to_shape=resize_to_shape,
                                                                                                                                                   face_square_expanded_resized=True)

                if profile_time:
                    get_modified_face_time = time.time()
                    get_modified_face_dur.append(get_modified_face_time - get_person_lm_time)    

                # Note the landmark coordinates w.r.t. the face_frame
                person_face_landmarks.append([video_face_base_name] + [list(l) for l in landmarks_in_face_square_expanded_resized])
    
                # Save the face image
                if save_faces:
                    imageio.imwrite(video_face_name, face_square_expanded_resized)
    
                if profile_time:
                    save_face_time = time.time()
                    save_face_dur.append(save_face_time - get_modified_face_time)

                # If save_with_blackened_mouths_and_polygon, combine the two
                if save_faces_combined_with_blackened_mouths_and_lip_polygons:
                    face_with_blackened_mouth_and_lip_polygons = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, landmarks_in_face_square_expanded_resized[48:68])
                    face_combined = np.hstack((face_square_expanded_resized, face_with_blackened_mouth_and_lip_polygons))
                    imageio.imwrite(video_face_combined_name, face_combined)
    
                if profile_time:
                    make_bmp_and_save_time = time.time()
                    make_bmp_and_save_dur.append(make_bmp_and_save_time - save_face_time)

            # Else if person's face is not present
            else:
               person_frame_landmarks.append([video_frame_name] + [])

            if profile_time:
                current_loop_end_time = time.time()
                loop_dur.append(current_loop_end_time - loop_end_time)
                loop_end_time = current_loop_end_time

    except KeyboardInterrupt:
        # Clean exit by saving face shapes and landmarks
        frame_number -= 1

    # Save the face shapes in frames, if not done already
    if detect_face_shapes:
        print("Saving", face_shapes_in_frames_all_persons_pkl_file_name)
        joblib.dump(face_shapes_in_frames_all_persons, face_shapes_in_frames_all_persons_pkl_file_name)

    # Save person frame landmarks
    print("Saving", landmarks_in_frames_person_txt_file_name)
    utils.write_landmarks_list_as_txt(landmarks_in_frames_person_txt_file_name, person_frame_landmarks)

    # Save person face landmarks
    print("Saving", landmarks_in_faces_person_txt_file_name)
    utils.write_landmarks_list_as_txt(landmarks_in_faces_person_txt_file_name, person_face_landmarks)

    if profile_time:
        save_stuff_time = time.time()

        init_dur = init_time - start_time
        save_frame_dur_avg = np.mean(save_frame_dur + [1e-8])
        detect_face_shapes_dur_avg = np.mean(detect_face_shapes_dur + [1e-8])
        get_person_lm_dur_avg = np.mean(get_person_lm_dur + [1e-8])
        get_modified_face_dur_avg = np.mean(get_modified_face_dur + [1e-8])
        save_face_dur_avg = np.mean(save_face_dur + [1e-8])
        make_bmp_and_save_dur_avg = np.mean(make_bmp_and_save_dur + [1e-8])
        loop_dur_avg = (loop_end_time - init_time)/(frame_number + 1)
        save_stuff_dur = save_stuff_time - loop_end_time
        print("init_dur                    : {0:.04f} seconds".format(init_dur))
        print("----------------------------------------------")
        print("save_frame_dur_avg          : {0:.04f} seconds".format(save_frame_dur_avg))
        print("detect_face_shapes_dur_avg  : {0:.04f} seconds".format(detect_face_shapes_dur_avg))
        print("get_person_lm_dur_avg       : {0:.04f} seconds".format(get_person_lm_dur_avg))
        print("get_modified_face_dur_avg   : {0:.04f} seconds".format(get_modified_face_dur_avg))
        print("save_face_dur_avg           : {0:.04f} seconds".format(save_face_dur_avg))
        print("make_bmp_and_save_dur_avg   : {0:.04f} seconds".format(make_bmp_and_save_dur_avg))
        print("----------------------------------------------")
        print("loop_dur_avg                : {0:.04f} seconds".format(loop_dur_avg))
        print("save_stuff_dur              : {0:.04f} seconds".format(save_stuff_dur))


def get_person_face_lm_from_face_shapes(frame, face_shapes, person_face_descriptor, dlib_facerec):
    '''Get the facial landmarks of a specific person
    http://dlib.net/face_recognition.py.html
    '''

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

