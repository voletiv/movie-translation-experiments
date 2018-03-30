import cv2
import imageio
import math
import numpy as np
import os
import subprocess
import sys
import tqdm

from exchange_dialogues_params import *

sys.path.append('../dataset_creation')
from movie_translation_data_creation_functions import *


config = MovieTranslationConfig()


def load_generator(model_path, verbose=False):
    if not os.path.exists(model_path):
        raise ValueError("[ERROR] model path does not exist! Given:", model_path)

    from keras.models import load_model
    if verbose:
        print("Loading model", model_path, "...")
    return load_model(model_path)


def exchange_dialogues(generator_model,
                       video1_language="telugu", video1_actor="Mahesh_Babu", video1_number=47,
                       video2_language="telugu", video2_actor="Mahesh_Babu", video2_number=89,
                       output_dir='.', verbose=False):

    # Generator model input shape
    _, generator_model_input_rows, generator_model_input_cols, _ = generator_model.layers[0].input_shape

    # Video 1 frames and landmarks
    if verbose:
        print("Getting video1 dir and landmarks")
    try:
        video1_frames_dir = get_video_frames_dir(video1_language, video1_actor, video1_number)
        
        # Read 2D landmarks detected using dlib (dlib.net)
        video1_2D_landmarks = read_landmarks(video1_language, video1_actor, video1_number, '2D_dlib')
        
       # Read 3D landmarks detected using face_alignment trained on LS3D-W (https://github.com/1adrianb/face-alignment)
        video1_3D_landmarks = read_landmarks(video1_language, video1_actor, video1_number, '3D')
    
    except ValueError as err:
        raise ValueError(err)

    video1_length = len(video1_2D_dlib_landmarks)

    # Video 2
    if verbose:
        print("Getting video2 dir and landmarks")
    try:
        video2_frames_dir = get_video_frames_dir(video2_language, video2_actor, video2_number)
        
        # Read 2D landmarks detected using dlib (dlib.net)
        video2_2D_landmarks = read_landmarks(video2_language, video2_actor, video2_number, '2D_dlib')

       # Read 3D landmarks detected using face_alignment trained on LS3D-W (https://github.com/1adrianb/face-alignment)
        video2_3D_landmarks = read_landmarks(video2_language, video2_actor, video2_number, '3D')

    except ValueError as err:
        raise ValueError(err)

    video2_length = len(video2_2D_landmarks)

    # Choose the smaller one as the target length, and choose those many central frames
    if verbose:
        print("Choosing smaller video")
    if video1_length < video2_length:
        if verbose:
            print("    video1 chosen")
        video_length = video1_length
        video1_frame_numbers = np.arange(video_length)
        video2_frame_numbers = np.arange((video2_length//2 - video_length//2), (video2_length//2 - video_length//2 + video_length))
        video2_2D_landmarks = video2_2D_landmarks[(video2_length//2 - video_length//2):(video2_length//2 - video_length//2 + video_length)]
        video2_3D_landmarks = video2_3D_landmarks[(video2_length//2 - video_length//2):(video2_length//2 - video_length//2 + video_length)]
    else:
        if verbose:
            print("    video2 chosen")
        video_length = video2_length
        video2_frame_numbers = np.arange(video_length)
        video1_frame_numbers = np.arange((video1_length//2 - video_length//2), (video1_length//2 - video_length//2 + video_length))
        video1_2D_landmarks = video1_2D_landmarks[(video1_length//2 - video_length//2):(video1_length//2 - video_length//2 + video_length)]
        video1_3D_landmarks = video1_3D_landmarks[(video1_length//2 - video_length//2):(video1_length//2 - video_length//2 + video_length)]

    # EXCHANGE DIALOGUES
    video1_frames_with_black_mouth_and_video2_lip_polygons = []
    video2_frames_with_black_mouth_and_video1_lip_polygons = []

    if video1_language != video2_language or video1_actor != video2_actor or video1_number != video2_number:
        process_video2 = True
    else:
        process_video2 = False

    # For each frame
    # read frame, blacken mouth, make new landmarks' polygon
    for i in tqdm.tqdm(range(video_length)):

        # Read video1 frame
        video1_frame_name = video1_actor + '_%04d_frame_%03d.png' % (video1_number, video1_frame_numbers[i])
        try:
            video1_frame = cv2.cvtColor(cv2.imread(os.path.join(video1_frames_dir, video1_frame_name)), cv2.COLOR_BGR2RGB)
        except:
            print("[ERROR]: Could not find", os.path.join(video1_frames_dir, video1_frame_name), "--- [SOLUTION] Retaining previous frame and landmarks")
            # Frame was not saved => landmarks were not detected
            # Retain prev video1_frame, save prev landmarks as this frame's landmarks
            video1_2D_landmarks[video1_frame_numbers[i]] = video1_2D_landmarks[video1_frame_numbers[i-1]]
            video1_3D_landmarks[video1_frame_numbers[i]] = video1_3D_landmarks[video1_frame_numbers[i-1]]

        if process_video2:
            # Read video2 frame
            video2_frame_name = video2_actor + '_%04d_frame_%03d.png' % (video2_number, video2_frame_numbers[i])
            try:
                video2_frame = cv2.cvtColor(cv2.imread(os.path.join(video2_frames_dir, video2_frame_name)), cv2.COLOR_BGR2RGB)
            except:
                print("[ERROR]: Could not find", os.path.join(video2_frames_dir, video2_frame_name), "--- [SOLUTION] Retaining previous frame and landmarks")
                # Frame was not saved => landmarks were not detected
                # Retain prev video2_frame, save prev landmarks as this frame's landmarks
                video2_2D_landmarks[video2_frame_numbers[i]] = video2_2D_landmarks[video2_frame_numbers[i-1]]
                video2_3D_landmarks[video2_frame_numbers[i]] = video2_3D_landmarks[video2_frame_numbers[i-1]]
        else:
            video2_frame = video1_frame

        # Get the frame's landmarks
        # Make 3D landmarks as x,y from dlib, and z from LS3D face-alignment
        video1_frame_3D_landmarks = np.hstack(( np.array(video1_2D_landmarks[video1_frame_numbers[i]][1:])[:, :2],
                                                np.array(video1_3D_landmarks[video1_frame_numbers[i]][1:])[:, 2].reshape(68, 1) ))
        if process_video2:
            video2_frame_3D_landmarks = np.hstack(( np.array(video2_2D_landmarks[video2_frame_numbers[i]][1:])[:, :2],
                                                    np.array(video2_3D_landmarks[video2_frame_numbers[i]][1:])[:, 2].reshape(68, 1) ))
        else:
            video2_frame_3D_landmarks = video1_frame_3D_landmarks

        # Saving some default previous landmarks
        if i == 0:
            prev_new_video1_lip_landmarks = video1_frame_3D_landmarks[48:68, :2]
            prev_new_video2_lip_landmarks = video2_frame_3D_landmarks[48:68, :2]

        # Exchange landmarks
        video1_3D_landmarks_tx_to_2, video2_3D_landmarks_tx_to_1 = exchange_3D_landmarks_using_3D_affine_tx(video1_frame_3D_landmarks, video2_frame_3D_landmarks,
                                                                                                            process_video2=process_video2, verbose=verbose)

        # If landmarks are not detected in the new frames, save as old frame's landmarks
        if video1_3D_landmarks_tx_to_2 is None:
            new_video1_lip_landmarks = prev_new_video1_lip_landmarks
        else:
            new_video1_lip_landmarks = video2_3D_landmarks_tx_to_1[48:68, :2]
        if process_video2:
            if video2_3D_landmarks_tx_to_2 is None:
                new_video2_lip_landmarks = prev_new_video2_lip_landmarks
            else:
                new_video2_lip_landmarks = video1_3D_landmarks_tx_to_2[48:68, :2]

        # Make frames with black mouth and polygon of landmarks
        video1_frame_with_black_mouth_and_video2_lip_polygons = make_black_mouth_and_lips_polygons(video1_frame, new_video1_lip_landmarks)
        if process_video2:
            video2_frame_with_black_mouth_and_video1_lip_polygons = make_black_mouth_and_lips_polygons(video2_frame, new_video2_lip_landmarks)

        # Resize frame to input_size of generator_model
        video1_frame_with_black_mouth_and_video2_lip_polygons_resized = cv2.resize(video1_frame_with_black_mouth_and_video2_lip_polygons, (generator_model_input_rows, generator_model_input_cols), interpolation=cv2.INTER_AREA)
        if process_video2:
            video2_frame_with_black_mouth_and_video1_lip_polygons_resized = cv2.resize(video2_frame_with_black_mouth_and_video1_lip_polygons, (generator_model_input_rows, generator_model_input_cols), interpolation=cv2.INTER_AREA)

        # Append frame to list
        video1_frames_with_black_mouth_and_video2_lip_polygons.append(video1_frame_with_black_mouth_and_video2_lip_polygons_resized)
        if process_video2:
            video2_frames_with_black_mouth_and_video1_lip_polygons.append(video2_frame_with_black_mouth_and_video1_lip_polygons_resized)

        # Save prev_new_video_frame_landmarks
        prev_new_video1_lip_landmarks = new_video1_lip_landmarks
        if process_video2:
            prev_new_video2_lip_landmarks = new_video2_lip_landmarks

    # Save black mouth polygons as mp4 with audio
    new_video1_file_name = video1_language + '_' + video1_actor + '_%04d' % video1_number + '_with_audio_of_' + video2_language + '_' + video2_actor + '_%04d' % video2_number + '_black_mouth_polygons.mp4'
    save_new_video_frames_with_old_audio_as_mp4(np.array([frame for frame in video1_frames_with_black_mouth_and_video2_lip_polygons]).astype('uint8'),
                                                audio_language=video2_language, audio_actor=video2_actor, audio_number=video2_number,
                                                output_dir=output_dir, file_name=new_video1_file_name, verbose=verbose)

    if process_video2:
        new_video2_file_name = video2_language + '_' + video2_actor + '_%04d' % video2_number + '_with_audio_of_' + video1_language + '_' + video1_actor + '_%04d' % video1_number + '_black_mouth_polygons.mp4'
        save_new_video_frames_with_old_audio_as_mp4(np.array([frame for frame in video2_frames_with_black_mouth_and_video1_lip_polygons]).astype('uint8'),
                                                    audio_language=video1_language, audio_actor=video1_actor, audio_number=video1_number,
                                                    output_dir=output_dir, file_name=new_video2_file_name, verbose=verbose)

    # Generate new frames
    if verbose:
        print("Generating new frames using Pix2Pix")
    new_video1_frames_generated = generator_model.predict(normalize_input_to_generator(video1_frames_with_black_mouth_and_video2_lip_polygons))
    if process_video2:
        new_video2_frames_generated = generator_model.predict(normalize_input_to_generator(video2_frames_with_black_mouth_and_video1_lip_polygons))

    # Rescale generated frames from -1->1 to 0->255
    new_video1_frames_generated = unnormalize_output_from_generator(new_video1_frames_generated)
    if process_video2:
        new_video2_frames_generated = unnormalize_output_from_generator(new_video2_frames_generated)
    else:
        new_video2_frames_generated = new_video1_frames_generated

    # Save as new mp4 with audio
    new_video1_file_name = video1_language + '_' + video1_actor + '_%04d' % video1_number + '_with_audio_of_' + video2_language + '_' + video2_actor + '_%04d' % video2_number + '.mp4'
    save_new_video_frames_with_old_audio_as_mp4(new_video1_frames_generated,
                                                audio_language=video2_language, audio_actor=video2_actor, audio_number=video2_number,
                                                output_dir=output_dir, file_name=new_video1_file_name, verbose=verbose)
    if process_video2:
        new_video2_file_name = video2_language + '_' + video2_actor + '_%04d' % video2_number + '_with_audio_of_' + video1_language + '_' + video1_actor + '_%04d' % video1_number + '.mp4'
        save_new_video_frames_with_old_audio_as_mp4(new_video2_frames_generated,
                                                    audio_language=video1_language, audio_actor=video1_actor, audio_number=video1_number,
                                                    output_dir=output_dir, file_name=new_video2_file_name, verbose=verbose)

    return new_video1_frames_generated, new_video2_frames_generated


#################################################
# DEPENDENT FUNCTIONS
#################################################


def get_video_frames_dir(language, actor, number):
    frames_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'frames', language, actor, actor + '_%04d' % number)
    if not os.path.exists(frames_dir):
        raise ValueError("[ERROR]: frames_dir", frames_dir, "does not exist!")
    else:
        return frames_dir


def read_landmarks(language, actor, number, read_2D_dlib_or_3D=''):
    '''
    language: e.g. 'telugu'
    actor: e.g. 'Mahesh_Babu'
    number: e.g. 3
    read_2D_dlib_or_3D = '' or '2D_dlib' or '3D'
    '''
    if read_2D_dlib_or_3D:
        actor_suffix = '_' + read_2D_dlib_or_3D
    landmarks_file = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'landmarks', language, actor + actor_suffix, actor + '_%04d' % number + "_landmarks.txt")
    if not os.path.exists(landmarks_file):
        raise ValueError("[ERROR] landmarks file does not exist! Given: " + landmarks_file)
    else:
        return read_landmarks_list_from_txt(landmarks_file)


def exchange_3D_landmarks_using_3D_affine_tx(video1_frame_3D_landmarks, video2_frame_3D_landmarks,
                                             process_video2=True, verbose=False):
    '''
    Exchange the landmarks of the two frames - estimate the affine 3D
    transformations between the 3D landmarks of the two frames (1->2 and 2->1),
    using only the first 36 landmarks, affine transform each frame's mouth
    landmarks using the Tx matrices, and return the new lip landmarks
    Note: Inputs are all 3D facial landmarks, but outputs are only lip landmarks
    INPUTS:
    video1_frame, video1_frame_landmarks, video2_frame, video2_frame_landmarks, process_video2
    face_alignment_object : an object of the FaceAlignment class to find landmarks
    OUTPUTS:
    new_video1_lip_landmarks, new_video2_lip_landmarks
    '''

    # 1 -> 2
    if verbose:
        print("Affine 3D Tx 1 -> 2")
    video1_3D_landmarks_tx_to_2 = affine_3D_tx_facial_landmarks_src_to_dst(video1_frame_3D_landmarks, video2_frame_3D_landmarks)

    # 2 -> 1
    if process_video2:
        if verbose:
            print("Affine 3D Tx 2 -> 1")
        video2_3D_landmarks_tx_to_1 = affine_3D_tx_facial_landmarks_src_to_dst(video2_frame_3D_landmarks, video1_frame_3D_landmarks)
    else:
        video2_3D_landmarks_tx_to_1 = None

    return video1_3D_landmarks_tx_to_2, video2_3D_landmarks_tx_to_1


def affine_3D_tx_facial_landmarks_src_to_dst(source_frame_3D_landmarks, target_frame_3D_landmarks):
    # Estimate Affine 3D transformation between the first 36 landmarks (jaw, eyebrows, eyes, nose bridge, nose base) from source to target
    retval, Rt_1_to_2, _ = cv2.estimateAffine3D(source_frame_3D_landmarks[:36], target_frame_3D_landmarks[:36])
    if retval is True:
        # Get the Affine transformed landmarks
        source_3D_landmarks_tx_to_target = np.dot( Rt_1_to_2, np.hstack(( source_frame_3D_landmarks, np.ones((68, 1)) )).T ).T
    else:
        source_3D_landmarks_tx_to_target = None
    return source_3D_landmarks_tx_to_target
    

def exchange_lip_landmarks_using_homography(video1_frame, video1_frame_landmarks,
                                            video2_frame, video2_frame_landmarks,
                                            using_dlib_or_face_alignment,
                                            dlib_detector=None, dlib_predictor=None, face_alignment_object=None,
                                            process_video2=True, verbose=False):
    '''
    Exchange the landmarks of the two frames - warp each frame to the other using homography,
    and return the respective new lip landmarks by detecting landmarks in each warped image
    Note: Inputs are all facial landmarks, but outputs are only lip landmarks
    INPUTS:
    face_alignment_object : an object of the FaceAlignment class to find landmarks
    video1_frame, video1_frame_landmarks, video2_frame, video2_frame_landmarks, process_video2
    OUTPUTS:
    new_video1_lip_landmarks, new_video2_lip_landmarks
    '''

    if using_dlib_or_face_alignment == 'dlib':
        if dlib_detector is None or dlib_predictor is None:
            print("\n\n[ERROR] Please provide dlib_detector and dlib_predictor! (Since you have chosen the option of 'dlib' in 'using_dlib_or_face_alignment')\n\n")
            return

    elif using_dlib_or_face_alignment == 'face_alignment':
        if face_alignment_object is None:
            print("\n\n[ERROR] Please provide face_alignment_object! (Since you have chosen the option of 'face_alignment' in 'using_dlib_or_face_alignment')\n\n")
            return

    _, video1_lip_landmarks_ur, video1_lip_landmarks_uc, video1_lip_landmarks_sr, video1_lip_landmarks_sc = normalize_lip_landmarks(video1_frame_landmarks[48:68, :2])
    _, video2_lip_landmarks_ur, video2_lip_landmarks_uc, video2_lip_landmarks_sr, video2_lip_landmarks_sc = normalize_lip_landmarks(video2_frame_landmarks[48:68, :2])

    # New video1 landmarks: landmarks of 2 -> landmarks of 1

    # Warp 2 to match 1's landmarks using Homography
    if verbose:
        print("exchange_landmarks: warping video2 to video1 using homography...")
    video2_frame_warped_to_1 = find_homography_warped_image(video2_frame, video2_frame_landmarks[:36, :2],
                                                            video1_frame_landmarks[:36, :2], image_size=video1_frame.shape[:2])

    # Get the warped image's landmarks
    if verbose:
        print("exchange_landmarks: finding new_video1_frame_landmarks of video2_frame_warped_to_1...")
    if using_dlib_or_face_alignment == 'dlib':
        new_video1_frame_landmarks = get_landmarks_using_dlib_detector_and_predictor(video2_frame_warped_to_1, dlib_detector, dlib_predictor)
    elif using_dlib_or_face_alignment == 'face_alignment':
        new_video1_frame_landmarks = get_landmarks_using_FaceAlignment(video2_frame_warped_to_1, face_alignment_object)

    # Align new lip landmarks with old lip landmarks' position and scale
    if new_video1_frame_landmarks is not None:
        if verbose:
            print("exchange_landmarks: finding new_video1_lip_landmarks by normalizing and unnormalizing new_video1_frame_landmarks...")
        new_video1_lip_landmarks = np.round(unnormalize_lip_landmarks(normalize_lip_landmarks(new_video1_frame_landmarks[48:68, :2])[0],
                                                                      video1_lip_landmarks_ur, video1_lip_landmarks_uc,
                                                                      video1_lip_landmarks_sr, video1_lip_landmarks_sc)).astype(int)
    else:
        print("exchange_landmarks: video2_frame_warped_to_1 has no landmarks!")
        new_video1_lip_landmarks = None

    # New video2 landmarks: landmarks of 1 -> landmarks of 2
    if process_video2:

        # Warp 1 to match 2's landmarks using Homography
        if verbose:
            print("exchange_landmarks: warping video1 to video2 using homography...")
        video1_frame_warped_to_2 = find_homography_warped_image(video1_frame, video1_frame_landmarks[:36, :2],
                                                                video2_frame_landmarks[:36, :2], image_size=video2_frame.shape[:2])

        # Get the warped image's landmarks
        if verbose:
            print("exchange_landmarks: finding new_video2_frame_landmarks of video1_frame_warped_to_2...")
        if using_dlib_or_face_alignment == 'dlib':
            new_video2_frame_landmarks = get_landmarks_using_dlib_detector_and_predictor(video1_frame_warped_to_2, dlib_detector, dlib_predictor)
        elif using_dlib_or_face_alignment == 'face_alignment':
            new_video2_frame_landmarks = get_landmarks_using_FaceAlignment(video1_frame_warped_to_2, face_alignment_object)

        # Align new lip landmarks with old lip landmarks' position adn scale
        if new_video2_frame_landmarks is not None:
            if verbose:
                print("exchange_landmarks: finding new_video2_lip_landmarks by normalizing and unnormalizing new_video1_frame_landmarks...")
            new_video2_lip_landmarks = np.round(unnormalize_lip_landmarks(normalize_lip_landmarks(new_video2_frame_landmarks[48:68, :2])[0],
                                                                          video2_lip_landmarks_ur, video2_lip_landmarks_uc,
                                                                          video2_lip_landmarks_sr, video2_lip_landmarks_sc)).astype(int)
        else:
            print("exchange_landmarks: video1_frame_warped_to_2 has no landmarks!")
            new_video2_lip_landmarks = None

    else:
        new_video2_lip_landmarks = None

    return new_video1_lip_landmarks, new_video2_lip_landmarks


def find_homography_warped_image(src_image, src_points, dst_points, image_size=(224, 224)):
    M, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(src_image, M, image_size)
    return warped_image


def normalize_input_to_generator(list_of_frames):
    return np.array(list_of_frames)/127. - 1


def unnormalize_output_from_generator(np_array_output_of_generator):
    return ((np_array_output_of_generator + 1)/2.*255).astype('uint8')


def save_new_video_frames_with_old_audio_as_mp4(frames,
                                                audio_language="telugu", audio_actor="Mahesh_Babu", audio_number=89,
                                                output_dir='.', file_name='new_video.mp4', verbose=False):

    # Save mp4 of frames
    if verbose:
        print("Writing frames as mp4")
    imageio.mimwrite('/tmp/video.mp4', frames , fps=24)

    # Extract audio from original files
    # orig_video, start_time, duration = get_metadata(language=audio_language, actor=audio_actor, number=audio_number)
    # command = ['ffmpeg', '-loglevel', 'warning', '-ss', start_time, '-i', orig_video, '-t', duration, '-y',
    #                '-vn', '-acodec', 'aac', '-strict', '-2', '/tmp/video_audio.aac']
    command = ['ffmpeg', '-loglevel', 'error',
               '-i', os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'videos', audio_language, audio_actor, audio_actor + '_%04d.mp4' % audio_number),
               '-vn', '-y', '-acodec', 'aac', '-strict', '-2', '/tmp/video_audio.aac']
    if verbose:
        print("Extracting audio from original files:", command)
    commandReturn = subprocess.call(command)    # subprocess.call returns 0 on successful run

    # Combine frames with audio
    if not os.path.exists(output_dir):
        print("Making", output_dir)
        os.makedirs(output_dir)

    command = ['ffmpeg', '-loglevel', 'error',
               '-i', '/tmp/video.mp4', '-i', '/tmp/video_audio.aac',
               '-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main', '-acodec', 'aac', '-strict', '-2',
               os.path.join(output_dir, file_name)]

    if verbose:
        print("Combining frames with audio:", command)
    commandReturn = subprocess.call(command)    # subprocess.call returns 0 on successful run

    # # Save npz
    # if verbose:
    #     print("Saving npz")
    # np.savez("exchanged_dialogues", new_video1=new_video1_frames_generated, new_video2=new_video2_frames_generated)

    # # Save GIF
    # if verbose:
    #     print("Saving gifs")
    # imageio.mimsave(os.path.join("video1.gif"), new_video1_frames_generated)
    # imageio.mimsave(os.path.join("video2.gif"), new_video2_frames_generated)


def get_metadata(language="telugu", actor="Mahesh_Babu", number=47):
    metadata_file = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "metadata", language, actor + '.txt')
    with open(metadata_file, 'r') as f:
        for l, line in enumerate(f):
            if l == number:
                metadata = line.strip().split()
                break
    return os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "youtube_videos", metadata[1] + '.mp4'), metadata[2], metadata[3]


def plot_landmarks(frame, landmarks):
    frame = np.array(frame)
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)
    plt.imshow(frame)
    plt.show()


def unrotate_lip_landmarks(lip_landmarks):
    # lip_landmarks = list(lip_landmarks)
    angle_rotated_by = math.atan((lip_landmarks[6][1] - lip_landmarks[0][1])/(lip_landmarks[6][0] - lip_landmarks[0][0]))
    rotated_lip_landmarks = rotate_points(lip_landmarks, lip_landmarks[0], -angle_rotated_by)
    return rotated_lip_landmarks, lip_landmarks[0], angle_rotated_by


def rotate_points(points, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    # When the points are row matrices, R is:
    R = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    return origin + np.dot(points-origin, R)


def normalize_lip_landmarks(lip_landmarks):
    ur, uc = lip_landmarks[0]
    sr, sc = lip_landmarks[:, 0].max() - lip_landmarks[:, 0].min(), lip_landmarks[:, 1].max() - lip_landmarks[:, 1].min()
    return (lip_landmarks - [ur, uc])/[sr, sc], ur, uc, sr, sc


def unnormalize_lip_landmarks(lip_landmarks, ur, uc, sr, sc):
    return lip_landmarks * [sr, sc] + [ur, uc]


def unrotate_lip_landmarks_point_by_point(lip_landmarks):
    lip_landmarks = list(lip_landmarks)
    angle_to_rotate_by = -math.atan((lip_landmarks[6][1] - lip_landmarks[0][1])/((224-lip_landmarks[6][0]) - (224-lip_landmarks[0][0])))
    rotated_lip_landmarks = [lip_landmarks[0]]
    for lip_landmark in lip_landmarks[1:]:
        [rot_x, rot_y] = rotate_point((lip_landmark[0], 224-lip_landmark[1]), (lip_landmarks[0][0], 224-lip_landmarks[0][1]), angle_to_rotate_by)
        rotated_lip_landmarks.append([int(round(rot_x)), int(round(224-rot_y))])
    return np.array(rotated_lip_landmarks), angle_to_rotate_by


def rotate_point(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]

