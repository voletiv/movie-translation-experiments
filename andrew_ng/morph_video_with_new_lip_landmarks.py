import argparse
import cv2
import glob
import imageio
import numpy as np
import os
import re
import sys
import time
import tqdm

from scipy import interpolate
from scipy.io import loadmat
from scipy.signal import medfilt
from skimage.transform import resize

import morph_video_config

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils


def get_closed_lip_cluster_center():
    lip_clusters = np.load(morph_video_config.LIP_CLUSTERS_FILE)
    return lip_clusters[morph_video_config.CLOSED_LIP_CLUSTER_INDEX].reshape(2, 20).transpose(1, 0)


def detect_no_voice_activity(audio_file, voice_activity_threshold, video_fps):
    from vad import VoiceActivityDetector
    if os.path.splitext(audio_file)[-1] != '.wav':
        ret = subprocess.call(['ffmpeg', '-loglevel', 'error', '-i', audio_file, '-vn', '-y', '-codec:a', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-f', 'wav', '%s.wav' % audio_file])
        audio_file = '%s.wav' % audio_file
    v = VoiceActivityDetector(audio_file, voice_activity_threshold)
    speech_detection = v.detect_speech()[:, 1]
    x = np.arange(0, len(speech_detection)*0.01, 0.01)
    f = interpolate.interp1d(x, speech_detection)
    xnew = np.arange(0, len(speech_detection)*0.01, 1/video_fps)
    ynew = np.round(f(xnew)).astype(int)
    return 1 - ynew


def interpolate_landmarks_to_new_fps(landmarks_in_frames, video_fps_old, video_fps_new):
    if video_fps_old == video_fps_new:
        return landmarks_in_frames
    x = np.arange(len(landmarks_in_frames))
    x_new = np.arange(0, len(landmarks_in_frames),
                      (len(landmarks_in_frames) - 1)/(len(landmarks_in_frames)/video_fps_old*video_fps_new - 1))
    x_new[x_new > x[-1]] = x[-1]
    landmarks_in_frames_new = np.zeros((len(x_new), *landmarks_in_frames.shape[1:]))
    for lm in range(landmarks_in_frames.shape[1]):
        for d in range(2):
            y = landmarks_in_frames[:, lm, d]
            f = interpolate.interp1d(x, y)
            y_new = f(x_new)
            landmarks_in_frames_new[:, lm, d] = y_new
    return landmarks_in_frames_new


def stabilize(landmarks):

    new_landmarks = []
    new_landmarks.append(landmarks[0])
    M = np.zeros((2, 3))

    for i in range(1, len(landmarks)):

        top_y = landmarks[i][:, 1].min()
        top_y_prev = new_landmarks[-1][:, 1].min()
        bottom_y = landmarks[i][48:68, 1].mean()
        bottom_y_prev = new_landmarks[-1][48:68, 1].mean()

        scale_y = (bottom_y_prev - top_y_prev) / (bottom_y - top_y)

        scaled_lm = landmarks[i]
        if scale_y > 1.05:
            scaled_lm[:, 1] = (scaled_lm[:, 1] - top_y)*scale_y + top_y

        new_landmarks.append(scaled_lm)
        '''
        source_landmarks = landmarks[i][[0, 8, 16], :].astype('float32')
        target_landmarks = new_landmarks[-1][[0, 8, 16], :].astype('float32')
        M = cv2.getAffineTransform(source_landmarks, target_landmarks)
        if M is None:
            M = prev_M
        else:
            if np.all(M == np.zeros((2, 3))):
                M = np.array([[1, 0, 0], [0, 1, 0]]).astype('float')
            prev_M = M
        new_landmarks.append(np.round( np.dot( M, np.hstack(( landmarks[i], np.ones((len(landmarks[i]), 1)) )).T ).T ))
        '''
    return np.array(new_landmarks).astype('float')


def read_video_landmarks(video_frames=None, # Either read landmarks for each frame
                         read_from_landmarks_file=True, video_landmarks_file=None, # Or, read from landmarks_file (if video_landmarks_file is not None)
                         video_file_name=None, landmarks_type='frames', video_fps=morph_video_config.ANDREW_NG_VIDEO_FPS, # Or, read from appropriate landmarks_file of video_file_name (if video_landmarks_file is None, and video_file_name is not None)
                         using_dlib_or_face_alignment='face_alignment',
                         dataset_dir=morph_video_config.dataset_dir, person=morph_video_config.person,
                         required_number=None, stabilize_landmarks=False,
                         save_landmarks=False, landmarks_file_name="blahblah.txt",
                         verbose=False):
    """
    Read landmarks
    1) from files with landmarks in full frames like /shared/fusor/home/voleti.vikram/ANDREW_NG/landmarks_in_frames_person/CV_01.C4W1L01_Computer_Vision_in_landmarks_frames_andrew_ng.txt,
    => read_from_landmarks_file=True, landmarks_type='frames', video_file_name is REQUIRED
    2) from files with landmarks in face images, like /shared/fusor/home/voleti.vikram/ANDREW_NG/landmarks_in_faces_person/CV_01.C4W1L01_Computer_Vision_in_landmarks_faces_andrew_ng.txt
    => read_from_landmarks_file=True, landmarks_type='faces', video_file_name is REQUIRED
    3) from the frames themselves (detect faces and predict landmarks using dlib)
    => read_from_landmarks_file=False, video_frames is REQUIRED
    """

    frames_with_no_landmarks = []

    # If read_from_landmarks_file, instead of detecting for every frame
    if read_from_landmarks_file:

        if verbose:
            print("read_video_landmarks: read_from_landmarks_file")

        if video_landmarks_file == None:

            # Find appropriate landmarks_file based on video_file_name
            if verbose:
                print("read_video_landmarks: read landmarks for video_file_name", video_file_name)
    
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
                raise ValueError("ERROR: landmarks_dir not not exist! Given:" + landmarks_dir)

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

        else:
            landmarks_file = video_landmarks_file

        if verbose:
            print("read_video_landmarks: Found landmarks file", landmarks_file)

        # Read all landmarks of all frames with person in video_file_name
        landmarks_full = utils.read_landmarks_list_from_txt(landmarks_file)

        # FROM frame
        if video_landmarks_file == None:
            # If landmarks_file is not mentioned, find the time_start_index (and maybe required_number) from video_file_name
            try:
                # Find time start frame number from video_file_name, if it exists
                time_start_index = re.search(r"[0-9][0-9][0-9][0-9][0-9][0-9]_to_[0-9][0-9][0-9][0-9][0-9][0-9]", video_file_name).start()
                time_start_hr = int(video_file_name[time_start_index:time_start_index+2])
                time_start_min = int(video_file_name[time_start_index+2:time_start_index+4])
                time_start_sec = int(video_file_name[time_start_index+4:time_start_index+6])
                time_start_frame = int(video_fps * (time_start_hr*3600 + time_start_min*60 + time_start_sec))
                if verbose:
                    print(video_file_name[time_start_index:time_start_index+15], time_start_hr, time_start_min, time_start_sec, time_start_frame)
            except:
                # Else, start from first frame
                time_start_frame = 0

        else:
            # If landmarks_file is specified, start from first frame in landmarks_file
            time_start_frame = 0

        if verbose:
            print("start_frame_number", time_start_frame)

        # If required number is not given, make it total length of video
        if required_number is None:
            if video_landmarks_file == None:
                try:
                    # Find time end frame number from video_file_name, if it exists
                    time_end_index = time_start_index + 10
                    time_end_hr = int(video_file_name[time_end_index:time_end_index+2])
                    time_end_min = int(video_file_name[time_end_index+2:time_end_index+4])
                    time_end_sec = int(video_file_name[time_end_index+4:time_end_index+6])
                    time_end_frame = int(video_fps * (time_end_hr*3600 + time_end_min*60 + time_end_sec))
                except:
                    # Else, end at last landmark frame
                    time_end_frame = int(os.path.splitext(landmarks_full[-1][0])[0].split("_")[-1])
            else:
                # Else, end at last landmark frame
                time_end_frame = int(os.path.splitext(landmarks_full[-1][0])[0].split("_")[-1])
            required_number = time_end_frame - time_start_frame + 1 

        if verbose:
            print("required_number", required_number, ";", time_start_frame, "to", time_start_frame+required_number, "; last_landmark_frame", int(os.path.splitext(landmarks_full[-1][0])[0].split("_")[-1]))

        # EXTRACT LANDMARKS
        landmarks = []

        # Find landmarks_full_index of start frame
        landmarks_full_index = 0
        while int(os.path.splitext(landmarks_full[landmarks_full_index][0])[0].split("_")[-1]) < time_start_frame:
            landmarks_full_index += 1

        for frame_number in range(time_start_frame, time_start_frame+required_number):
            if landmarks_full_index < len(landmarks_full):
                landmarks_frame_number = int(os.path.splitext(landmarks_full[landmarks_full_index][0])[0].split("_")[-1])
                landmarks_of_lm_frame_number = landmarks_full[landmarks_full_index]
            if verbose:
                print("frame_number", frame_number, "; landmarks_frame_number", landmarks_frame_number)
            if landmarks_frame_number == frame_number:
                frames_with_no_landmarks.append(0)
                landmarks.append(landmarks_of_lm_frame_number)
                landmarks_full_index += 1
            else:
                # If landmarks_frame_number > frame_number, no landmarks were detected for current frame_number
                frames_with_no_landmarks.append(1)
                landmarks.append(landmarks_of_lm_frame_number)

        # Save only landmarks (without frame number)
        landmarks = [lms[1:] for lms in landmarks]

    # Else, detect landmarks using dlib
    else:

        if video_frames == None:
            raise ValueError("ERROR: read_video_frame_landmarks: video_frames needs to be given, since read_from_landmarks_file=False!")

        print("read_video_frame_landmarks: detecting faces and predicting landmarks in every frame using " + str(using_dlib_or_face_alignment) + "...")

        if using_dlib_or_face_alignment == 'dlib':
            dlib_face_detector, dlib_shape_predictor = utils.load_dlib_detector_and_predictor(verbose=verbose)
            landmarks = []
            for f, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):
                if using_dlib_or_face_alignment == 'dlib':
                    landmarks_in_frame = utils.get_landmarks_using_dlib_detector_and_predictor(frame, dlib_face_detector, dlib_shape_predictor)
                if landmarks_in_frame is not None:
                    landmarks.append(landmarks_in_frame[:, :2])
                    frames_with_no_landmarks.append(0)
                else:
                    frames_with_no_landmarks.append(1)
                    if f > 0:
                        landmarks.append(landmarks[-1])
                    else:
                        landmarks.append(np.zeros((68, 2)))

        elif using_dlib_or_face_alignment == 'face_alignment':
            tmp_video_frames_npy_file = '/tmp/video_frames.npy'
            np.save(tmp_video_frames_npy_file, video_frames)
            subprocess.call([str(sys.executable), 'detect_landmarks_using_FaceAlignment.py', tmp_video_frames_npy_file])
            os.remove(tmp_video_frames_npy_file)
            my_video_landmarks = np.load('/tmp/my_video_landmarks.npz')
            landmarks = my_video_landmarks['landmarks']
            frames_with_no_landmarks = my_video_landmarks['frames_with_no_landmarks']
        else:
            raise ValueError("read_video_frame_landmarks:'using_dlib_or_face_alignment' can only be 'dlib' or 'face_alignment', got " + str(using_dlib_or_face_alignment))


    # TODO
    # Make landmarks_list for landmarks.txt
    # Save landmarks
    # utils.write_landmarks_list_as_txt(os.path.join(landmarks_3D_in_frames_dir, video_file_name + "_landmarks_in_frames.txt"), landmarks_in_frames_list)

    # Convert to float array
    landmarks = np.array(landmarks).astype('float')

    # Use medan filter to smoothen the output
    # landmarks = medfilt(landmarks, (13, 1, 1))

    # Stabilize landmarks
    if stabilize_landmarks:
        landmarks = stabilize(landmarks)

    return landmarks, frames_with_no_landmarks


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


def affine_transform_landmarks(source_landmarks, target_landmarks, fullAffine=False, prev_M=np.zeros((2, 3))):
    M = cv2.estimateRigidTransform(source_landmarks.astype('float'), target_landmarks.astype('float'), fullAffine)
    if M is None:
        M = prev_M
    target_landmarks_tx_from_source = np.round( np.dot( M, np.hstack(( source_landmarks, np.ones((len(source_landmarks), 1)) )).T ).T ).astype('int')
    return target_landmarks_tx_from_source, M


def transform_landmarks_by_upper_lips(source_lip_landmarks, target_lip_landmarks):

    source_lip_landmarks = source_lip_landmarks.astype('float')
    target_lip_landmarks = target_lip_landmarks.astype('float')

    # Centroid
    source_lips_centroid = np.mean(source_lip_landmarks, axis=0)
    target_lips_centroid = np.mean(target_lip_landmarks, axis=0)

    # Upper lips
    source_upper_lips = source_lip_landmarks[:7]
    target_upper_lips = target_lip_landmarks[:7]

    # Upper lips centroid
    source_upper_lips_centroid = np.mean(source_upper_lips, axis=0)
    target_upper_lips_centroid = np.mean(target_upper_lips, axis=0)

    # Centre both mouths
    source_mouth_centred = source_lip_landmarks - source_lips_centroid
    target_mouth_centred = target_lip_landmarks - target_lips_centroid

    # Scale
    scale_x = (np.max(target_mouth_centred[:, 0]) - np.min(target_mouth_centred[:, 0]))/(np.max(source_mouth_centred[:, 0]) - np.min(source_mouth_centred[:, 0]))
    scale_y = (np.max(target_mouth_centred[:, 1]) - np.min(target_mouth_centred[:, 1]))/(np.max(source_mouth_centred[:, 1]) - np.min(source_mouth_centred[:, 1]))
    source_mouth_centred_scaled = source_mouth_centred
    source_mouth_centred_scaled[:, 0] *= scale_x
    source_mouth_centred_scaled[:, 1] *= scale_y

    # Move upper lip centroid
    translation = target_upper_lips_centroid - source_upper_lips_centroid
    new_mouth_landmarks = source_mouth_centred_scaled + source_lips_centroid - source_upper_lips_centroid + translation

    return np.round(new_mouth_landmarks).astype('int')


def transform_landmarks_by_mouth_centroid_and_scales(source_lip_landmarks, target_lip_landmarks):

    source_lip_landmarks = source_lip_landmarks.astype('float')
    target_lip_landmarks = target_lip_landmarks.astype('float')
    
    # Mouth left corner
    ml_source = source_lip_landmarks[0]
    ml_target = target_lip_landmarks[0]

    # Mouth right corner
    mr_source = source_lip_landmarks[6]
    mr_target = target_lip_landmarks[6]

    # Mouth top
    mt_source = source_lip_landmarks[3]
    mt_target = target_lip_landmarks[3]

    # Mouth bottom
    mb_source = source_lip_landmarks[9]
    mb_target = target_lip_landmarks[9]

    # Centroid
    mouth_centroid_source = (ml_source + mr_source + mt_source + mb_source)/4
    mouth_centroid_target = (ml_target + mr_target + mt_target + mb_target)/4

    # Centre both mouths
    mouth_centred_source = source_lip_landmarks - mouth_centroid_source
    mouth_centred_target = target_lip_landmarks - mouth_centroid_target

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


def transform_landmarks_by_mouth_centroid_and_scale_x(source_lip_landmarks, target_lip_landmarks):

    source_lip_landmarks = source_lip_landmarks.astype('float')
    target_lip_landmarks = target_lip_landmarks.astype('float')

    # Mouth left corner
    ml_source = source_lip_landmarks[0]
    ml_target = target_lip_landmarks[0]

    # Mouth right corner
    mr_source = source_lip_landmarks[6]
    mr_target = target_lip_landmarks[6]

    # Mouth top
    mt_source = source_lip_landmarks[3]
    mt_target = target_lip_landmarks[3]

    # Mouth bottom
    mb_source = source_lip_landmarks[9]
    mb_target = target_lip_landmarks[9]

    # Centroid
    mouth_centroid_source = (ml_source + mr_source + mt_source + mb_source)/4
    mouth_centroid_target = (ml_target + mr_target + mt_target + mb_target)/4

    # Centre both mouths
    mouth_centred_source = source_lip_landmarks - mouth_centroid_source
    mouth_centred_target = target_lip_landmarks - mouth_centroid_target

    # Calculate scale_x
    scale_x = (mr_target - ml_target)[0]/(mr_source - ml_source)[0]
    # print(scale_x)

    # Scale the source centred landmarks
    # mouth_centred_source = mouth_centred_source * scale_x
    mouth_centred_source[:, 0] = mouth_centred_source[:, 0] * scale_x
    mouth_centred_source[:, 1] = mouth_centred_source[:, 1] * scale_x * 1.4

    # Centre it to the target centre
    new_mouth_landmarks = mouth_centred_source + mouth_centroid_target

    return np.round(new_mouth_landmarks).astype('int')


def transform_landmarks_by_mouth_centroid_and_memorize_scale_x(source_lip_landmarks, target_lip_landmarks, scale_x):

    source_lip_landmarks = source_lip_landmarks.astype('float')
    target_lip_landmarks = target_lip_landmarks.astype('float')

    # Mouth left corner
    ml_source = source_lip_landmarks[0]
    ml_target = target_lip_landmarks[0]

    # Mouth right corner
    mr_source = source_lip_landmarks[6]
    mr_target = target_lip_landmarks[6]

    # Mouth top
    mt_source = source_lip_landmarks[3]
    mt_target = target_lip_landmarks[3]

    # Mouth bottom
    mb_source = source_lip_landmarks[9]
    mb_target = target_lip_landmarks[9]

    # Centroid
    mouth_centroid_source = (ml_source + mr_source + mt_source + mb_source)/4
    mouth_centroid_target = (ml_target + mr_target + mt_target + mb_target)/4

    # Centre both mouths
    mouth_centred_source = source_lip_landmarks - mouth_centroid_source
    mouth_centred_target = target_lip_landmarks - mouth_centroid_target

    # Calculate scale_x
    if scale_x is None:
        scale_x = (mr_target - ml_target)[0]/(mr_source - ml_source)[0]
    # print(scale_x)

    # Scale the source centred landmarks
    # mouth_centred_source = mouth_centred_source * scale_x
    mouth_centred_source[:, 0] = mouth_centred_source[:, 0] * scale_x
    mouth_centred_source[:12, 1] = mouth_centred_source[:12, 1] * scale_x * 1.4
    mouth_centred_source[12:, 1] = mouth_centred_source[12:, 1] * scale_x * 1.6

    # Centre it to the target centre
    new_mouth_landmarks = mouth_centred_source + mouth_centroid_target

    return np.round(new_mouth_landmarks).astype('int'), scale_x


def tmp_morph_video_with_new_lip_landmarks(generator_model_name, target_video_file, target_audio_file, lip_landmarks_mat_file, lip_landmarks_fps, output_video_name,
                                           target_video_landmarks_file=None, save_making=True, save_generated_video=True, stabilize_landmarks=False,
                                           detect_landmarks_in_video=False, using_dlib_or_face_alignment='face_alignment',
                                           replace_closed_mouth=False, voice_activity_threshold=0.6, lm_prepend_time_in_ms=200,
                                           constant_face=False, use_identity=False, ffmpeg_overwrite=False, verbose=False):

    # Read predicted lip landmarks    
    mat = loadmat(lip_landmarks_mat_file)
    new_lip_landmarks = mat['y_pred']

    # Call the actual function
    morph_video_with_new_lip_landmarks(generator_model_name=generator_model_name, target_video_file=target_video_file,
                                       target_audio_file=target_audio_file, new_lip_landmarks=new_lip_landmarks, lip_landmarks_fps=lip_landmarks_fps,
                                       output_video_name=output_video_name, target_video_landmarks_file=target_video_landmarks_file, save_making=save_making,
                                       save_generated_video=save_generated_video, stabilize_landmarks=stabilize_landmarks,
                                       detect_landmarks_in_video=detect_landmarks_in_video, using_dlib_or_face_alignment=using_dlib_or_face_alignment,
                                       replace_closed_mouth=replace_closed_mouth, voice_activity_threshold=voice_activity_threshold, lm_prepend_time_in_ms=lm_prepend_time_in_ms,
                                       constant_face=constant_face, use_identity=use_identity, ffmpeg_overwrite=ffmpeg_overwrite, verbose=verbose)


def morph_video_with_new_lip_landmarks(generator_model_name, target_video_file, target_audio_file, new_lip_landmarks, lip_landmarks_fps=morph_video_config.generated_lip_landmarks_fps,
                                       output_video_name=None, target_video_landmarks_file=None, save_making=True, save_generated_video=True, stabilize_landmarks=False,
                                       detect_landmarks_in_video=False, using_dlib_or_face_alignment='face_alignment',
                                       replace_closed_mouth=False, voice_activity_threshold=0.6, lm_prepend_time_in_ms=200,
                                       constant_face=False, use_identity=False, ffmpeg_overwrite=False, verbose=False):

    # If constant_face, update output_video_name
    output_video_name = os.path.splitext(output_video_name)[0] + "_constant_face" + os.path.splitext(output_video_name)[1]

    # If use_identity, update output_video_name
    output_video_name = os.path.splitext(output_video_name)[0] + "_withIdentity" + os.path.splitext(output_video_name)[1]

    # Read target video
    target_video_reader = imageio.get_reader(target_video_file)
    target_video_fps = target_video_reader.get_meta_data()['fps']
    if verbose:
        print("target_video_length:", len(target_video_reader), "; target_video_fps:", target_video_fps)

    # Note source landmarks
    source_lip_landmarks = new_lip_landmarks

    # Change fps of landmarks to target_fps from lip_landmarks_fps
    source_lip_landmarks = interpolate_landmarks_to_new_fps(source_lip_landmarks, lip_landmarks_fps, target_video_fps)

    # Prepend 5 landmark frames (because they were generated by a TIME-DELAYED LSTM..)
    number_of_frames_to_prepend = (target_video_fps * lm_prepend_time_in_ms / 1000)
    source_lip_landmarks = np.concatenate((np.repeat(np.expand_dims(source_lip_landmarks[0], axis=0), number_of_frames_to_prepend, axis=0), source_lip_landmarks), axis=0)

    num_of_frames = len(source_lip_landmarks)
    if verbose:
        print("Number of frames:", num_of_frames)

    # Make lip closures when no speech
    if replace_closed_mouth:
        if verbose:
            print("Replacing closed mouth with cluster center")
        if target_audio_file is None:
            print("ERROR: target_audio_file is None! Not replacing closed mouth with cluster center.")
        else:
            lip_landmarks_to_replace = detect_no_voice_activity(target_audio_file, voice_activity_threshold, target_video_fps)
            print(lip_landmarks_to_replace)
            closed_lip_landmarks = get_closed_lip_cluster_center()
            source_lip_landmarks[lip_landmarks_to_replace] = closed_lip_landmarks

    # Read as many target video frames as source landmarks
    target_video_frames = []

    if verbose:
        print("Reading target video frames...")

    # If contant_face, read only 20 frames and choose the last
    if constant_face:
        for f, frame in enumerate(target_video_reader):
            target_video_frames.append(frame)
            if f == 20:
                break

        target_video_frames = np.tile(target_video_frames[-1], (num_of_frames, 1, 1, 1))
    
    else:
        for f, frame in enumerate(target_video_reader):
            target_video_frames.append(frame)
            if f+1 == num_of_frames:
                break

    # Read target_video_file's frame landmarks
    target_all_landmarks_in_frames, frames_with_no_landmarks = read_video_landmarks(video_frames=target_video_frames, video_file_name=target_video_file,
                                                                                    read_from_landmarks_file=(not detect_landmarks_in_video), using_dlib_or_face_alignment=using_dlib_or_face_alignment,
                                                                                    video_landmarks_file=target_video_landmarks_file,
                                                                                    landmarks_type='frames', required_number=num_of_frames, video_fps=target_video_fps,
                                                                                    stabilize_landmarks=stabilize_landmarks, verbose=verbose)

    # If constant_face, choose the landmarks of the only face considered
    if constant_face:
        target_all_landmarks_in_frames = np.tile(target_all_landmarks_in_frames[20], (len(target_all_landmarks_in_frames), 1, 1))

    # LOAD GENERATOR
    # Generator model
    generator_model = utils.load_generator(generator_model_name, verbose=verbose)

    # Generator model input shape
    _, generator_model_input_rows, generator_model_input_cols, _ = generator_model.layers[0].input_shape
    if verbose:
        print("generator_model input shape:", (generator_model_input_rows, generator_model_input_cols))

    if use_identity:
        generator_model_input_cols = generator_model_input_cols // 2

    assert (generator_model_input_rows == generator_model_input_cols), "Please ensure gen_model has correct input size! Found " + \
        str(generator_model_input_rows) + "x" + str(generator_model_input_cols) + ". use_identity=" + str(use_identity)

    # Identity face
    if use_identity:
        # 1) Take the middle frame as the identity frame
        identity_frame_number = len(target_all_landmarks_in_frames)//2
        # If the middle frame has no landmarks, increment frame number until you find a frame with lm
        while frames_with_no_landmarks[identity_frame_number] == 1 and identity_frame_number >= 0:
            print("Searching for frame with landmarks in lesser half...")
            identity_frame_number -= 1
        if identity_frame_number == -1:
            identity_frame_number = len(target_all_landmarks_in_frames)//2
            while frames_with_no_landmarks[identity_frame_number] == 1:
                print("Searching for frame with landmarks in greater half...")
                identity_frame_number += 1
        identity_frame = target_video_frames[identity_frame_number]
        identity_frame_landmarks = target_all_landmarks_in_frames[identity_frame_number]
        # 2) Get the face - squared, expanded, resized
        identity_face, _, _, _ = utils.get_square_expand_resize_face_and_modify_landmarks(np.array(identity_frame),
                                                                                          identity_frame_landmarks,
                                                                                          resize_to_shape=(generator_model_input_rows, generator_model_input_cols),
                                                                                          face_square_expanded_resized=True)

    # Make new images of faces with black mouth polygons
    face_rect_in_frames = []
    face_original_sizes = []
    faces_original = []
    faces_with_black_mouth_polygons = []
    making_frames = []

    # Affine Tx from source to target
    M = np.zeros((2, 3))
    # First scale_y; in case transform_landmarks_by_mouth_centroid_and_scale_x_memorize_scale_y is being used
    scale_x = None

    """
    target_video_frames = np.array([target_video_frames[93]] * len(target_video_frames))
    target_all_landmarks_in_frames[:] = target_all_landmarks_in_frames[93]
    closed_lip_landmarks = get_closed_lip_cluster_center()
    source_lip_landmarks[:] = closed_lip_landmarks
    # """

    if verbose:
        print("Making new images of faces with black mouth polygons...")

    for (target_frame, target_all_landmarks_in_frame, source_lip_landmarks_in_frame, use_original_frame) in tqdm.tqdm(zip(target_video_frames,
                                                                                                                          target_all_landmarks_in_frames,
                                                                                                                          source_lip_landmarks,
                                                                                                                          frames_with_no_landmarks),
                                                                                                                      total=num_of_frames):

        # Get the face - squared, expanded, resized
        face_square_expanded_resized, \
            landmarks_in_face_square_expanded_resized, \
            face_rect_in_frame, face_original_size = utils.get_square_expand_resize_face_and_modify_landmarks(np.array(target_frame),
                                                                                                              target_all_landmarks_in_frame,
                                                                                                              resize_to_shape=(generator_model_input_rows, generator_model_input_cols),
                                                                                                              face_square_expanded_resized=True)

        # Note face rect in frame, face original size
        faces_original.append(face_square_expanded_resized)
        face_rect_in_frames.append(face_rect_in_frame)
        face_original_sizes.append(face_original_size)

        # Tx source lip landmarks to good target video position, etc.
        target_lip_landmarks_in_frame = np.array(landmarks_in_face_square_expanded_resized[48:68])
        # target_lip_landmarks_tx_from_source, M = affine_transform_landmarks(source_lip_landmarks_in_frame, target_lip_landmarks_in_frame, fullAffine=False, prev_M=M)
        # target_lip_landmarks_tx_from_source = transform_landmarks_by_upper_lips(source_lip_landmarks_in_frame, target_lip_landmarks_in_frame)
        # target_lip_landmarks_tx_from_source = transform_landmarks_by_mouth_centroid_and_scale_x(source_lip_landmarks_in_frame, target_lip_landmarks_in_frame)
        target_lip_landmarks_tx_from_source, scale_x = transform_landmarks_by_mouth_centroid_and_memorize_scale_x(source_lip_landmarks_in_frame, target_lip_landmarks_in_frame, scale_x)

        # Make face with black mouth polygon
        face_with_bmp = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, target_lip_landmarks_tx_from_source)
        if save_making:
            face_with_original_bmp = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, landmarks_in_face_square_expanded_resized[48:68])
            making_frame = np.hstack((face_square_expanded_resized, face_with_original_bmp))
            making_frame = np.vstack(( making_frame, np.hstack((np.zeros(face_with_bmp.shape), face_with_bmp)) ))
            making_frames.append(making_frame)
        if use_identity:
            face_with_bmp = np.concatenate((face_with_bmp, identity_face), axis=1)
        faces_with_black_mouth_polygons.append(face_with_bmp)

    # GENERATE FACES USING PIX2PIX
    if save_generated_video:

        # Predict new faces using generator
        if verbose:
            print("Generating new faces using faces_with_bmp...")

        # Predict in batches
        new_faces = []
        gen_input_faces = []
        batch_size = 8
        num_of_batches = int(np.ceil(len(faces_with_black_mouth_polygons)/batch_size))
        for batch in tqdm.tqdm(range(num_of_batches)):
            faces_with_bmp_batch = faces_with_black_mouth_polygons[batch*batch_size:(batch+1)*batch_size]
            for face in faces_with_bmp_batch:
                gen_input_faces.append(face)
            new_faces_batch = utils.unnormalize_output_from_generator(generator_model.predict(utils.normalize_input_to_generator(faces_with_bmp_batch)))
            for new_face in new_faces_batch:
                if use_identity:
                    new_faces.append(new_face[:, :generator_model_input_cols])
                else:
                    new_faces.append(new_face)
 

        # print("Saving input faces as", os.path.splitext(output_video_name)[0]+"_input_faces.mp4")
        # utils.save_new_video_frames_with_target_audio_as_mp4(gen_input_faces, target_video_fps, target_audio_file,
        #                                                      output_file_name=os.path.splitext(output_video_name)[0]+"_input_faces.mp4",
        #                                                      overwrite=ffmpeg_overwrite, verbose=verbose)

        if save_making:
            for i in range(len(making_frames)):
                # if frames_with_no_landmarks[i]:
                #     making_frames[i][-generator_model_input_rows:, :generator_model_input_cols] = faces_original[i]
                # else:
                making_frames[i][-generator_model_input_rows:, :generator_model_input_cols] = new_faces[i]
   
        # Reintegrate generated faces into frames
        if verbose:
            print("Reintegrating generated faces into frames...")

        new_frames = list(target_video_frames)
        for i, (new_frame, new_face, face_original_size, face_rect_in_frame, use_original_frame) in tqdm.tqdm(enumerate(zip(new_frames, new_faces, face_original_sizes, face_rect_in_frames, frames_with_no_landmarks)),
                                                                                                           total=num_of_frames):
            if not use_original_frame:
                new_face_resized = np.round(resize(new_face, face_original_size, mode='reflect', preserve_range=True)).astype('uint8')
                new_frame[face_rect_in_frame[1]:face_rect_in_frame[3], face_rect_in_frame[0]:face_rect_in_frame[2]] = new_face_resized
            else:
                if verbose:
                    print("Copying original frame", i)

        # Write new video
        print("Saving new frames as", output_video_name)
        utils.save_new_video_frames_with_target_audio_as_mp4(new_frames, target_video_fps, target_audio_file,
                                                             output_file_name=output_video_name,
                                                             overwrite=ffmpeg_overwrite, verbose=verbose)

    if save_making:
        print("Saving making video as", os.path.splitext(output_video_name)[0] + '_making.mp4')
        resized_making_frames = []
        for frame in making_frames:
            resized_making_frames.append(np.round(resize(frame, (256, 256), mode='reflect', preserve_range=True)).astype('uint8'))
        utils.save_new_video_frames_with_target_audio_as_mp4(np.round(resized_making_frames).astype('uint8'), target_video_fps, target_audio_file,
                                                             output_file_name=os.path.splitext(output_video_name)[0] + '_making.mp4',
                                                             overwrite=ffmpeg_overwrite, verbose=verbose)



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
    parser.add_argument('--detect_landmarks_in_video', '-dlm', action="store_true")
    parser.add_argument('--using_dlib_or_face_alignment', type=str, choices=["dlib", "face_alignment"], default="face_alignment", help="Choose dlib or face_alignment to detect landmarks, IF detect_landmarks_in_video is True")
    parser.add_argument('--target_video_landmarks_file', '-t', type=str, default=None, help="landmarks file of target video; eg. /shared/fusor/home/voleti.vikram/ANDREW_NG_CLIPS/landmarks_in_frames_person/CV_01_C4W1L01_000003_to_000045/")
    parser.add_argument('--lip_landmarks_mat_file', '-l', type=str, help="Predicted and target lip landmarks; eg. /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/generated_hindi_landmarks/output5/generated_lip_landmarks.mat")
    parser.add_argument('--lip_landmarks_fps', '-lfps', type=float, default=25, help="FPS of lip landmarks generated")
    parser.add_argument('--output_video_name', '-o', type=str, default=None, help="Name of output video; def: <input_video_name>_with_lips_of_<lip_landmarks_mat>.mp4")
    parser.add_argument('--generator_model_name', '-g', type=str, default=morph_video_config.generator_model, help="Path to the generator model to be used; eg.: /shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/models/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5")
    parser.add_argument('--save_making', '-m', action="store_true")
    parser.add_argument('--dont_save_generated_video', '-d', action="store_true")
    parser.add_argument('--stabilize_landmarks', '-s', action="store_true")
    parser.add_argument('--replace_closed_mouth', '-r', action="store_true")
    parser.add_argument('--voice_activity_threshold', '-vadthresh', type=float, default=0.6, help="threshold [0 - 1] for voice activity detection; energy > thresh => voice")
    parser.add_argument('--lm_prepend_time_in_ms', '-lmptime', type=float, default=200, help="Time (in ms) to add landmarks at start due to delay in TIME-DELAYED LSTM")
    parser.add_argument('--constant_face', '-cf', action="store_true")
    parser.add_argument('--ffmpeg_overwrite', '-y', action="store_true")
    parser.add_argument('--use_identity', '-i', action="store_true")
    parser.add_argument('--verbose', '-v', action="store_true")
    args = parser.parse_args()

    # EXAMPLE: python morph_video_with_new_lip_landmarks.py /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4 -a /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/Obama/ouput_new5.aac -l /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/generated_hindi_landmarks/output5/generated_lip_landmarks.mat -b -f -c -v

    # EXAMPLE: python /shared/fusor/home/voleti.vikram/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4 -a /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/genhindi_erated_landmarks/test_wav/CV_05_C4W1L05_000001_to_000011.wav -l /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/genhindi_erated_landmarks/CV_05_C4W1L05_000001_to_000011_generated_lip_landmarks.mat -o /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/visdub_hindi/CV_05_C4W1L05_000001_to_000011_with_generated_lip_landmarks_upper_lip.mp4 -v

    # EXAMPLE: python /home/voleti.vikram/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4 -t /home/voleti.vikram/ANDREW_NG_CLIPS/landmarks_in_frames_person/CV_01_C4W1L01_000003_to_000045_CV_01_C4W1L01_000003_to_000045_landmarks_in_frames_andrew_ng.txt -a /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045_hindi_abhishek.wav -l /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045_hindi_abhishek_generated_lip_landmarks.mat -o /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045_hindi_abhishek.mp4 -r -m -v -y

    # Assert args
    assert_args(args)

    if args.verbose:
        print(args)

    try:

        # Run
        tmp_morph_video_with_new_lip_landmarks(args.generator_model_name,
                                               args.target_video_file, args.target_audio_file,
                                               args.lip_landmarks_mat_file, args.lip_landmarks_fps,
                                               args.output_video_name,
                                               target_video_landmarks_file=args.target_video_landmarks_file,
                                               save_making=args.save_making,
                                               save_generated_video=args.save_generated_video,
                                               stabilize_landmarks=args.stabilize_landmarks,
                                               detect_landmarks_in_video=args.detect_landmarks_in_video,
                                               using_dlib_or_face_alignment=args.using_dlib_or_face_alignment,
                                               replace_closed_mouth=args.replace_closed_mouth,
                                               voice_activity_threshold=args.voice_activity_threshold,
                                               lm_prepend_time_in_ms=args.lm_prepend_time_in_ms,
                                               constant_face=args.constant_face,
                                               ffmpeg_overwrite=args.ffmpeg_overwrite,
                                               use_identity=args.use_identity,
                                               verbose=args.verbose)

    except ValueError as e:
        print(e)

    except KeyboardInterrupt:
        print("Ctrl+C was pressed! Exiting.")

