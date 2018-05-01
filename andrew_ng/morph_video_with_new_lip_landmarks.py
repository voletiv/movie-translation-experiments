import argparse
import cv2
import glob
import imageio
import numpy as np
import os
import sys
import time
import tqdm

from scipy.io import loadmat
from skimage.transform import resize

import morph_video_config

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.append(ROOT_DIR)
import utils


def read_video_landmarks(video_file_name=None, video_frames=None,
                         read_from_landmarks_file=True, landmarks_type='frames',
                         dataset_dir=morph_video_config.dataset_dir, person=morph_video_config.person,
                         required_number=None, verbose=False):
    """
    Read landmarks
    1) from files with landmarks in full frames like /shared/fusor/home/voleti.vikram/ANDREW_NG/landmarks_in_frames_person/CV_01.C4W1L01_Computer_Vision_in_frames_andrew_ng.txt,
    => read_from_landmarks_file=True, landmarks_type='frames', video_file_name is REQUIRED
    2) from files with landmarks in face images, like shared/fusor/home/voleti.vikram/ANDREW_NG/landmarks_in_faces_person/CV_01.C4W1L01_Computer_Vision_in_faces_andrew_ng.txt
    => read_from_landmarks_file=True, landmarks_type='faces', video_file_name is REQUIRED
    3) from the frames themselves (detect faces and predict landmarks using dlib)
    => read_from_landmarks_file=False, video_frames is REQUIRED
    """

    if read_from_landmarks_file:

        if verbose:
            print("read_video_landmarks: read_from_landmarks_file")

        if video_file_name == None:
            raise ValueError("ERROR: read_video_frame_landmarks: video_file_name needs to be given, since read_from_landmarks_file=True!")

        if not os.path.exists(dataset_dir):
            raise ValueError("ERROR: dataset_dir does not exist! Given:" + dataset_dir)
   
        if landmarks_type == 'frames': 
            landmarks_dir = os.path.join(dataset_dir, 'landmarks_in_frames_person')
        elif landmarks_type == 'faces':
            landmarks_dir = os.path.join(dataset_dir, 'landmarks_in_faces_person')    
        else:
            raise ValueError("ERROR: landmarks_type can only be 'frames' or 'faces'! Given:" + landmarks_type)

        if not os.path.exists(landmarks_dir):
            raise ValueError("ERROR: landmarks_in_frames_person_dir not not exist! Given:" + landmarks_in_frames_person_dir)
    
        # Read all landmarks files
        landmarks_files = sorted(glob.glob(os.path.join(landmarks_dir, '*.txt')))
    
        # Get the right landmark_file by checking for video_file_name_checks in each landmark_file
        video_file_name_checks = os.path.basename(video_file_name).split('_')[:3] + [person]
        for landmarks_file in landmarks_files:
            this_one = True
            for video_file_name_check in video_file_name_checks:
                if video_file_name_check in landmarks_file:
                    this_one = this_one & True
                else:
                    this_one = this_one & False
            if this_one:
                break

        if not this_one:
            raise ValueError("ERROR: could not find any landmarks file for the video_file_name" + video_file_name)

        if verbose:
            print("read_video_landmarks: Found landmarks file", landmarks_file)

        # Read all landmarks of all frames of video_file_name
        landmarks_full = utils.read_landmarks_list_from_txt(landmarks_file)

        # Note only those landmarks of the required number of frames
        if required_number is not None:
            landmarks_full = landmarks_full[:required_number]

        # Save only mouth landmarks
        landmarks = np.array([lms[1:] for lms in landmarks_full]).astype('float')

        return landmarks

    else:

        if video_frames == None:
            raise ValueError("ERROR: read_video_frame_landmarks: video_frames needs to be given, since read_from_landmarks_file=False!")

        print("read_video_frame_landmarks: detecting faces and predicting landmarks in every frame...")

        dlib_face_detector, dlib_shape_predictor = load_dlib_detector_and_predictor(verbose=verbose)

        landmarks = []
        for frame in tqdm.tqdm(video_frames):
            landmarks_in_frame = get_landmarks_using_dlib_detector_and_predictor(frame, dlib_face_detector, dlib_shape_predictor)
            landmarks.append(landmarks_in_frame[1:])

        return np.array(landmarks)


def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
    # Denormalize the points
    kp_dn = N * kp_features_mouth
    # Add the tilt
    x, y = kp_dn[:, 0], kp_dn[:, 1]
    c, s = np.cos(tilt), np.sin(tilt)
    x_dash, y_dash = x*c + y*s, -x*s + y*c
    kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1,1))))
    # Shift to the mean
    kp = kp_tilt + mean
    return kp


def affine_transform_landmarks_abhishek(source_landmarks, target_landmarks):
    source_landmarks_tx_to_target = []
    for idx, (source_landmarks_per_frame, t) in enumerate(zip(source_landmarks, target_landmarks)):
        target_lip_landmarks_normalized, target_N, target_tilt, target_mean, target_all_landmarks_normalized, target_all_landmarks = k[0], k[1], k[2], k[3], k[4], k[5]
        kps = getOriginalKeypoints(source_landmarks_per_frame, target_N, target_tilt, target_mean)
        source_landmarks_mapped_to_target.append(kps)
    return source_landmarks_tx_to_target


def affine_transform_landmarks(source_landmarks, target_landmarks, fullAffine=True, prev_M=np.zeros((2, 3))):
    M = cv2.estimateRigidTransform(source_landmarks.astype('float'), target_landmarks.astype('float'), fullAffine)
    if M is None:
        M = prev_M
    target_landmarks_tx_from_source = np.round( np.dot( M, np.hstack(( source_landmarks, np.ones((len(source_landmarks), 1)) )).T ).T ).astype('int')
    return target_landmarks_tx_from_source, M


def simple_transform_landmarks(source_landmarks, target_landmarks):

    source_landmarks = source_landmarks.astype('float')
    target_landmarks = target_landmarks.astype('float')
    
    # Mouth left corner
    ml_source = source_landmarks[0]
    ml_target = target_landmarks[0]

    # Mouth right corner
    mr_source = source_landmarks[6]
    mr_target = target_landmarks[6]

    # Mouth top
    mt_source = source_landmarks[3]
    mt_target = target_landmarks[3]

    # Mouth bottom
    mb_source = source_landmarks[9]
    mb_target = target_landmarks[9]

    # Centroid
    mouth_centroid_source = (ml_source + mr_source + mt_source + mb_source)/4
    mouth_centroid_target = (ml_target + mr_target + mt_target + mb_target)/4

    # Centre both mouths
    mouth_centred_source = source_landmarks - mouth_centroid_source
    mouth_centred_target = target_landmarks - mouth_centroid_target

    # Calculate scales
    scale_x = (mr_target - ml_target)[0]/(mr_source - ml_source)[0]
    scale_y = (mt_target - mb_target)[1]/(mt_source - mb_source)[1]
    # print(scale_x, scale_y)

    # Scale the source centred landmarks
    mouth_centred_source[:, 0] = mouth_centred_source[:, 0] * scale_x
    mouth_centred_source[:, 1] = mouth_centred_source[:, 1] * scale_y

    # Centre it to the target centre
    new_mouth_landmarks = mouth_centred_source + mouth_centroid_target

    return np.round(new_mouth_landmarks).astype('int')


def tmp_morph_video_with_new_lip_landmarks(generator_model, target_video_file, target_audio_file, lip_landmarks_mat_file, output_video_name,
                                           save_faces_with_black_mouth_polygons=False, save_generated_faces=False,
                                           save_both_faces_with_bmp=False, save_generated_video=True, verbose=False):

    # Read predicted lip landmarks    
    mat = loadmat(lip_landmarks_mat_file)
    new_lip_landmarks = mat['y_pred']

    # Call the actual function
    morph_video_with_new_lip_landmarks(generator_model=generator_model, target_video_file=target_video_file,
                                       target_audio_file=target_audio_file, new_lip_landmarks=new_lip_landmarks, output_video_name=output_video_name,
                                       save_faces_with_black_mouth_polygons=save_faces_with_black_mouth_polygons, save_generated_faces=save_generated_faces,
                                       save_both_faces_with_bmp=save_both_faces_with_bmp, save_generated_video=save_generated_video, verbose=verbose)


def morph_video_with_new_lip_landmarks(generator_model, target_video_file, target_audio_file, new_lip_landmarks, output_video_name,
                                       save_faces_with_black_mouth_polygons=False, save_generated_faces=False,
                                       save_both_faces_with_bmp=False, save_generated_video=True, verbose=False):

    # Generator model input shape
    _, generator_model_input_rows, generator_model_input_cols, _ = generator_model.layers[0].input_shape
    if verbose:
        print("generator_model input shape:", (generator_model_input_rows, generator_model_input_cols))

    # Read target video
    target_video_reader = imageio.get_reader(target_video_file)
    target_video_fps = target_video_reader.get_meta_data()['fps']
    if verbose:
        print("target_video_fps:", target_video_fps)

    # Note source landmarks
    source_lip_landmarks = new_lip_landmarks
    num_of_frames = len(source_lip_landmarks)
    if verbose:
        print("Number of frames:", num_of_frames)

    # Read as many target video frames as source landmarks
    target_video_frames = []

    if verbose:
        print("Reading target video frames...")

    for f, frame in enumerate(target_video_reader):
        target_video_frames.append(frame)
        if f+1 == num_of_frames:
            break

    # Read target_video_file's frame landmarks
    target_all_landmarks_in_frames = read_video_landmarks(video_file_name=target_video_file, read_from_landmarks_file=True, landmarks_type='frames', required_number=num_of_frames, verbose=verbose)

    # Make new images of faces with black mouth polygons
    face_rect_in_frames = []
    face_original_sizes = []
    both_faces_with_bmp = []
    faces_with_black_mouth_polygons = []
    M = np.zeros((2, 3))

    if verbose:
        print("Making new images of faces with black mouth polygons...")

    for f, (target_frame, target_all_landmarks_in_frame, source_lip_landmarks_in_frame) in enumerate(tqdm.tqdm(zip(target_video_frames, target_all_landmarks_in_frames, source_lip_landmarks), total=num_of_frames)):

        # Get the face - squared, expanded, resized
        face_square_expanded_resized, \
            landmarks_in_face_square_expanded_resized, \
            face_rect_in_frame, face_original_size = utils.get_square_expand_resize_face_and_modify_landmarks(np.array(target_frame),
                                                                                                              target_all_landmarks_in_frame,
                                                                                                              resize_to_shape=(generator_model_input_rows, generator_model_input_cols),
                                                                                                              face_square_expanded_resized=True)

        # Note face rect in frame, face original size
        face_rect_in_frames.append(face_rect_in_frame)
        face_original_sizes.append(face_original_size)

        # Tx source lip landmarks to good target video position, etc.
        target_lip_landmarks_in_frame = np.array(landmarks_in_face_square_expanded_resized[48:68])
        # target_lip_landmarks_tx_from_source, M = affine_transform_landmarks(source_lip_landmarks_in_frame, target_lip_landmarks_in_frame, fullAffine=True, prev_M=M)
        target_lip_landmarks_tx_from_source = simple_transform_landmarks(source_lip_landmarks_in_frame, target_lip_landmarks_in_frame)

        # Make face with black mouth polygon
        face_with_bmp = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, target_lip_landmarks_tx_from_source)
        faces_with_black_mouth_polygons.append(face_with_bmp)
        if save_both_faces_with_bmp:
            face_with_original_bmp = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, landmarks_in_face_square_expanded_resized[48:68])
            both_face_with_bmp = np.hstack((face_with_original_bmp, face_with_bmp))
            both_faces_with_bmp.append(both_face_with_bmp)

    if save_faces_with_black_mouth_polygons:
        faces_with_bmp_output_video_name = os.path.splitext(output_video_name)[0] + '_faces_with_bmp.mp4'
        print("Saving faces with black mouth polygons as", faces_with_bmp_output_video_name)
        utils.save_new_video_frames_with_target_audio_as_mp4(faces_with_black_mouth_polygons, target_video_fps, target_audio_file, output_file_name=faces_with_bmp_output_video_name, verbose=verbose)

    if save_both_faces_with_bmp:
        both_faces_with_bmp_output_video_name = os.path.splitext(output_video_name)[0] + '_both_faces_with_bmp.mp4'
        print("Saving both faces with black mouth polygons as", both_faces_with_bmp_output_video_name)
        utils.save_new_video_frames_with_target_audio_as_mp4(both_faces_with_bmp, target_video_fps, target_audio_file, output_file_name=both_faces_with_bmp_output_video_name, verbose=verbose)

    if save_generated_video:

        # Predict new faces using generator
        if verbose:
            print("Generating new faces using faces_with_bmp...")

        # Predict in batches
        new_faces = []
        batch_size = 8
        num_of_batches = int(np.ceil(len(faces_with_black_mouth_polygons)/batch_size))
        for batch in range(num_of_batches):
            faces_with_bmp_batch = faces_with_black_mouth_polygons[batch*batch_size:(batch+1)*batch_size]
            new_faces_batch = utils.unnormalize_output_from_generator(generator_model.predict(utils.normalize_input_to_generator(faces_with_bmp_batch)))
            for new_face in new_faces_batch:
                new_faces.append(new_face)

        # new_faces = utils.unnormalize_output_from_generator(generator_model.predict(utils.normalize_input_to_generator(faces_with_black_mouth_polygons)))

        if save_generated_faces:
            faces_output_video_name = os.path.splitext(output_video_name)[0] + '_faces.mp4'
            print("Saving new faces as", faces_output_video_name)
            utils.save_new_video_frames_with_target_audio_as_mp4(new_faces, target_video_fps, target_audio_file, output_file_name=faces_output_video_name, verbose=verbose)
   
        # Reintegrate generated faces into frames
        if verbose:
            print("Reintegrating generated faces into frames...")

        new_frames = list(target_video_frames)
        for (new_frame, new_face, face_original_size, face_rect_in_frame) in tqdm.tqdm(zip(new_frames, new_faces, face_original_sizes, face_rect_in_frames), total=num_of_frames):
            new_face_resized = np.round(resize(new_face, face_original_size, mode='reflect', preserve_range=True)).astype('uint8')
            new_frame[face_rect_in_frame[1]:face_rect_in_frame[3], face_rect_in_frame[0]:face_rect_in_frame[2]] = new_face_resized

        # Write new video
        print("Saving new frames as", output_video_name)
        utils.save_new_video_frames_with_target_audio_as_mp4(new_frames, target_video_fps, target_audio_file, output_file_name=output_video_name, verbose=verbose)


def assert_args(args):
    try:
        # Assert target_video_file exists
        assert os.path.exists(args.target_video_file), ("Target video file does not exist! Given: " + args.target_video_file)
        # Assert target_audio_file exists
        if args.target_audio_file is not None:
            assert os.path.exists(args.target_audio_file), ("Target audio file does not exist! Given: " + args.target_audio_file)
        # Assert lip_landmarks_mat_file
        assert os.path.exists(args.lip_landmarks_mat_file), ("Lip landmarks file does not exist! Given: " + args.lip_landmarks_mat_file)
        # Assert generator model exists
        assert os.path.exists(args.generator_model_name), ("Generator model does not exist! Given: " + args.generator_model_name)
         # Set output video name (if not done already)
        if args.output_video_name is None:
            args.output_video_name = os.path.splitext(args.target_video_file)[0] + '_with_lips_of_' + os.path.splitext(os.path.basename(args.lip_landmarks_mat_file))[0] + '_mat.mp4'
        # Saving generated video or not
        if args.dont_save_generated_video:
            args.save_generated_video = False
        else:
            args.save_generated_video = True
    except AssertionError as error:
        print('\nERROR:\n', error, '\n')
        print("Exiting.\n")
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Morph lips in input video with target lip landmarks')
    parser.add_argument('target_video_file', type=str, help="target video (.mp4); eg. /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4")
    parser.add_argument('--target_audio_file', '-a', type=str, default=None, help="target audio; eg. /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/Obama/ouput_new5.mp3")
    parser.add_argument('--lip_landmarks_mat_file', '-l',type=str, help="Predicted and target lip landmarks; eg. /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/generated_hindi_landmarks/output5/generated_lip_landmarks.mat")
    parser.add_argument('--output_video_name', '-o', type=str, default=None, help="Name of output video; def: <input_video_name>_with_lips_of_<lip_landmarks_mat>.mp4")
    parser.add_argument('--generator_model_name', '-g', type=str, default=morph_video_config.generator_model, help="Path to the generator model to be used; eg.: /shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/models/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5")
    parser.add_argument('--save_faces_with_black_mouth_polygons', '-b', action="store_true")
    parser.add_argument('--save_generated_faces', '-f', action="store_true")
    parser.add_argument('--save_both_faces_with_bmp', '-c', action="store_true")
    parser.add_argument('--dont_save_generated_video', '-d', action="store_true")
    parser.add_argument('--verbose', '-v', action="store_true")
    args = parser.parse_args()

    # EXAMPLE: python morph_video_with_new_lip_landmarks.py /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4 -a /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/Obama/ouput_new5.aac -l /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/generated_hindi_landmarks/output5/generated_lip_landmarks.mat -b -f -c -v

    # Assert args
    assert_args(args)

    # Generator
    generator_model = utils.load_generator(args.generator_model_name, verbose=args.verbose)

    try:

        # Run
        tmp_morph_video_with_new_lip_landmarks(generator_model,
                                               args.target_video_file, args.target_audio_file, args.lip_landmarks_mat_file, args.output_video_name,
                                               save_faces_with_black_mouth_polygons=args.save_faces_with_black_mouth_polygons,
                                               save_generated_faces=args.save_generated_faces,
                                               save_both_faces_with_bmp=args.save_both_faces_with_bmp,
                                               save_generated_video=args.save_generated_video,
                                               verbose=args.verbose)

    except ValueError as e:
        print(e)

    except KeyboardInterrupt:
        print("Ctrl+C was pressed! Exiting.")
