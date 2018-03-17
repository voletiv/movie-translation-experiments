import cv2
import dlib
import imageio
import math
import numpy as np
import os
import subprocess
import tqdm

from exchange_dialogues_params import *

config = MovieTranslationConfig()


def load_generator(model_path):
    from keras.models import load_model
    return load_model(model_path)


def exchange_dialogues(generator_model,
                       video1_language="telugu", video1_actor="Mahesh_Babu", video1_number=47,
                       video2_language="telugu", video2_actor="Mahesh_Babu", video2_number=89,
                       output_dir='.', verbose=False):

    # Generator model input shape
    _, generator_model_input_rows, generator_model_input_cols, _ = generator_model.layers[0].input_shape

    # Video 1
    if verbose:
        print("Getting video1 dir and landmarks")
    try:
        video1_frames_dir = get_video_frames_dir(video1_language, video1_actor, video1_number)
        video1_landmarks = get_landmarks(video1_language, video1_actor, video1_number)
    except ValueError as err:
        raise ValueError(err)

    video1_length = len(video1_landmarks)

    # Video 2
    if verbose:
        print("Getting video2 dir and landmarks")
    try:
        video2_frames_dir = get_video_frames_dir(video2_language, video2_actor, video2_number)
        video2_landmarks = get_landmarks(video2_language, video2_actor, video2_number)
    except ValueError as err:
        raise ValueError(err)

    video2_length = len(video2_landmarks)

    # Choose the smaller one as the target length, and choose those many central frames
    if verbose:
        print("Choosing smaller video")
    if video1_length < video2_length:
        if verbose:
            print("    video1 chosen")
        video_length = video1_length
        video1_frame_numbers = np.arange(video1_length)
        video2_landmarks = video2_landmarks[(video2_length//2 - video1_length//2):(video2_length//2 - video1_length//2 + video1_length)]
        video2_frame_numbers = np.arange((video2_length//2 - video1_length//2), (video2_length//2 - video1_length//2 + video1_length))
    else:
        if verbose:
            print("    video2 chosen")
        video_length = video2_length
        video1_landmarks = video1_landmarks[(video1_length//2 - video2_length//2):(video1_length//2 - video2_length//2 + video2_length)]
        video1_frame_numbers = np.arange((video1_length//2 - video2_length//2), (video1_length//2 - video2_length//2 + video2_length))
        video2_frame_numbers = np.arange(video2_length)

    # EXCHANGE DIALOGUES
    video1_frames_with_black_mouth_and_video2_lip_polygons = []
    video2_frames_with_black_mouth_and_video1_lip_polygons = []

    # For each frame
    # read frame, blacken mouth, make new landmarks' polygon
    for i in tqdm.tqdm(range(video_length)):

        # Read video1 frame
        video1_frame_name = video1_actor + '_%04d_frame_%03d.png' % (video1_number, video1_frame_numbers[i])
        try:
            video1_frame = cv2.cvtColor(cv2.imread(os.path.join(video1_frames_dir, video1_frame_name)), cv2.COLOR_BGR2RGB)
        except:
            print("[ERROR]: Could not find", os.path.join(video1_frames_dir, video1_frame_name), "--- [SOLUTION] Retaining previous frame and landmarks")
            video1_landmarks[video1_frame_numbers[i]] = video1_landmarks[video1_frame_numbers[i-1]]

        # Read video2 frame
        video2_frame_name = video2_actor + '_%04d_frame_%03d.png' % (video2_number, video2_frame_numbers[i])
        try:
            video2_frame = cv2.cvtColor(cv2.imread(os.path.join(video2_frames_dir, video2_frame_name)), cv2.COLOR_BGR2RGB)
        except:
            print("[ERROR]: Could not find", os.path.join(video2_frames_dir, video2_frame_name), "--- [SOLUTION] Retaining previous frame and landmarks")
            video2_landmarks[video2_frame_numbers[i]] = video2_landmarks[video2_frame_numbers[i-1]]

        # Get the landmarks
        video1_frame_lip_landmarks = np.array(video1_landmarks[video1_frame_numbers[i]][1:][48:68])
        video2_frame_lip_landmarks = np.array(video2_landmarks[video2_frame_numbers[i]][1:][48:68])

        # Exchange landmarks
        new_video1_frame_lip_landmarks, new_video2_frame_lip_landmarks = exchange_landmarks(video1_frame_lip_landmarks, video2_frame_lip_landmarks)

        # Make frames with black mouth and polygon of landmarks
        video1_frame_with_black_mouth_and_video2_lip_polygons = make_black_mouth_and_lips_polygons(video1_frame, new_video1_frame_lip_landmarks)
        video2_frame_with_black_mouth_and_video1_lip_polygons = make_black_mouth_and_lips_polygons(video2_frame, new_video2_frame_lip_landmarks)

        # Resize frame to input_size of generator_model
        video1_frame_with_black_mouth_and_video2_lip_polygons_resized = cv2.resize(video1_frame_with_black_mouth_and_video2_lip_polygons, (generator_model_input_rows, generator_model_input_cols), interpolation=cv2.INTER_AREA)
        video2_frame_with_black_mouth_and_video1_lip_polygons_resized = cv2.resize(video2_frame_with_black_mouth_and_video1_lip_polygons, (generator_model_input_rows, generator_model_input_cols), interpolation=cv2.INTER_AREA)

        video1_frames_with_black_mouth_and_video2_lip_polygons.append(video1_frame_with_black_mouth_and_video2_lip_polygons_resized)
        video2_frames_with_black_mouth_and_video1_lip_polygons.append(video2_frame_with_black_mouth_and_video1_lip_polygons_resized)

    # Save black mouth polygons as mp4 with audio
    new_video1_file_name = video1_language + '_' + video1_actor + '_%04d' % video1_number + '_with_audio_of_' + video2_language + '_' + video2_actor + '_%04d' % video2_number + '_black_mouth_polygons.mp4'
    save_new_video_frames_with_old_audio_as_mp4(np.array([frame for frame in video1_frames_with_black_mouth_and_video2_lip_polygons]).astype('uint8'),
                                                audio_language=video2_language, audio_actor=video2_actor, audio_number=video2_number,
                                                output_dir=output_dir, file_name=new_video1_file_name, verbose=verbose)

    if video1_language != video2_language or video1_actor != video2_actor or video1_number != video2_number:
        process_video2 = True
    else:
        process_video2 = False

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


def get_landmarks(language, actor, number):
    landmarks_file = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'landmarks', language, actor, actor + '_%04d' % number + "_landmarks.txt")
    if not os.path.exists(landmarks_file):
        raiseValueError("[ERROR]: landmarks file", landmarks_file, "does not exist!")
    else:
        return read_landmarks_list_from_txt(landmarks_file)


def read_landmarks_list_from_txt(path):
    landmarks_list = []
    translate_table = dict((ord(char), None) for char in '[],')
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split(" [")
            landmarks_list.append([row[0]] + [[int(e.split(" ")[0].translate(translate_table)), int(e.split(" ")[1].translate(translate_table))] for e in row[1:]])
    return landmarks_list


def exchange_landmarks(video1_frame_lip_landmarks, video2_frame_lip_landmarks):

    # Unrotate both frames' lip landmarks
    video1_frame_lip_landmarks_unrotated, video1_landmarks_origin, angle_video1_landmarks_rotated_by = unrotate_lip_landmarks(video1_frame_lip_landmarks)
    video2_frame_lip_landmarks_unrotated, video2_landmarks_origin, angle_video2_landmarks_rotated_by = unrotate_lip_landmarks(video2_frame_lip_landmarks)

    # Normalize both frames' rotated lip landmarks
    video1_frame_lip_landmarks_unrotated_normalized, video1_ur, video1_uc, video1_sr, video1_sc = normalize_lip_landmarks(video1_frame_lip_landmarks_unrotated)
    video2_frame_lip_landmarks_unrotated_normalized, video2_ur, video2_uc, video2_sr, video2_sc = normalize_lip_landmarks(video2_frame_lip_landmarks_unrotated)

    # Make new lip landmarks by unnormalizing and then rotating
    new_video1_frame_lip_landmarks = np.round(rotate_points(unnormalize_lip_landmarks(video2_frame_lip_landmarks_unrotated_normalized,
                                                                                      video1_ur, video1_uc, video1_sr, video1_sc),
                                                            video1_landmarks_origin, angle_video1_landmarks_rotated_by)).astype('int')
    new_video2_frame_lip_landmarks = np.round(rotate_points(unnormalize_lip_landmarks(video1_frame_lip_landmarks_unrotated_normalized,
                                                                                      video2_ur, video2_uc, video2_sr, video2_sc),
                                                            video2_landmarks_origin, angle_video2_landmarks_rotated_by)).astype('int')

    return new_video1_frame_lip_landmarks, new_video2_frame_lip_landmarks


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


def plot_landmarks(frame, landmarks):
    frame = np.array(frame)
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)
    plt.imshow(frame)
    plt.show()


def plot_lip_landmarks(lip_landmarks, frame=None, video=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if video:
        plt.ion()
        fig.show()
        fig.canvas.draw()
    if frame is None:
        frame = np.zeros((224, 224))
    else:
        frame = np.array(frame)
    for l, lip_landmark in enumerate(lip_landmarks):
        frame[int(lip_landmark[1])-2:int(lip_landmark[1])+2, int(lip_landmark[0])-2:int(lip_landmark[0])+2] = 1
        ax.imshow(frame)
        if video:
            ax.set_title(str(l))
            fig.canvas.draw()
    if not video:
        plt.show()


def make_black_mouth_and_lips_polygons(frame, lip_landmarks):

        # Find mouth bounding box
        mouth_rect = dlib.rectangle(int(np.min(lip_landmarks[:, 0])), int(np.min(lip_landmarks[:, 1])), int(np.max(lip_landmarks[:, 0])), int(np.max(lip_landmarks[:, 1])))

        # Expand mouth bounding box
        mouth_rect_expanded = expand_rect(mouth_rect, scale_w=1.2, scale_h=1.8, frame_shape=(224, 224))

        # Make new frame for blackened mouth and lip polygons
        frame_with_blackened_mouth_and_lip_polygons = np.array(frame)

        # Blacken (expanded) mouth in frame
        frame_with_blackened_mouth_and_lip_polygons[mouth_rect_expanded.top():mouth_rect_expanded.bottom(),
                                                    mouth_rect_expanded.left():mouth_rect_expanded.right()] = 0

        # Draw lips polygon in frame
        frame_with_blackened_mouth_and_lip_polygons = cv2.drawContours(frame_with_blackened_mouth_and_lip_polygons,
                                                                       [lip_landmarks[:12], lip_landmarks[12:]], -1, (255, 255, 255))

        return frame_with_blackened_mouth_and_lip_polygons


def expand_rect(rect, scale=None, scale_w=1.5, scale_h=1.5, frame_shape=(256, 256)):
    if scale is not None:
        scale_w = scale
        scale_h = scale
    # dlib.rectangle
    if type(rect) == dlib.rectangle:
        x = rect.left()
        y = rect.top()
        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()
    else:
        # Rect: (x, y, w, h)
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
    new_w = int(w * scale_w)
    new_h = int(h * scale_h)
    new_x = max(0, min(frame_shape[1] - w, x - int((new_w - w) / 2)))
    new_y = max(0, min(frame_shape[0] - h, y - int((new_h - h) / 2)))
    # w = min(w, frame_shape[1] - x)
    # h = min(h, frame_shape[0] - y)
    if type(rect) == dlib.rectangle:
        return dlib.rectangle(new_x, new_y, new_x + new_w, new_y + new_h)
    else:
        return [new_x, new_y, new_w, new_h]


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
    return os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "in_progress", metadata[1] + '.mp4'), metadata[2], metadata[3]


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

