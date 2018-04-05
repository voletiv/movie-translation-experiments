import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from config import *


config = MovieTranslationConfig()


def load_dlib_detector_and_predictor(verbose=False):
    import dlib

    try:
        dlib_detector = dlib.get_frontal_face_detector()
        dlib_predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return dlib_detector, dlib_predictor

    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("\n\nERROR: Wrong Shape Predictor .dat file path - " + \
            config.SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)\n\n")


def load_face_alignment_object(enable_cuda=False, flip_input=False, use_cnn_face_detector=False, verbose=False):
    # Check https://github.com/1adrianb/face-alignment for installation instructions
    if verbose:
        print("Loading FaceAlignment object ...")
    import face_alignment
    return face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                        enable_cuda=enable_cuda,
                                        flip_input=flip_input,
                                        use_cnn_face_detector=use_cnn_face_detector)


def read_metadata(metadata_txt_file):
    d = []
    with open(metadata_txt_file, 'r') as f:
        for line in f:
            d.append(line.split())
    return d


def get_landmarks_using_FaceAlignment(frame, face_alignment_object):
    landmarks = face_alignment_object.get_landmarks(frame)
    if landmarks is not None:
        return np.round(landmarks[0]).astype('int')
    else:
        return None


def get_landmarks_using_dlib_detector_and_predictor(frame, detector, predictor):
    # Landmarks Coords: ------> x (cols)
    #                  |
    #                  |
    #                  v
    #                  y
    #                (rows)
    faces = detector(frame, 1)
    if len(faces) > 0:
        # TODO: Use VGG-Face to verify face
        # Choose first face
        face = faces[0]
        # Detect landmarks
        shape = predictor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), face)
        landmarks = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
        return np.round(landmarks).astype('int')
    else:
        return None


def make_black_mouth_and_lips_polygons(frame, mouth_landmarks):

        # Find mouth bounding box
        mouth_rect = [int(np.min(mouth_landmarks[:, 0])), int(np.min(mouth_landmarks[:, 1])), int(np.max(mouth_landmarks[:, 0])), int(np.max(mouth_landmarks[:, 1]))]

        # Expand mouth bounding box
        mouth_rect_expanded = expand_rect(mouth_rect, scale_w=1.2, scale_h=1.8, frame_shape=(224, 224))

        # Make new frame for blackened mouth and lip polygons
        frame_with_blackened_mouth_and_lip_polygons = np.array(frame)

        # Blacken (expanded) mouth in frame
        frame_with_blackened_mouth_and_lip_polygons[mouth_rect_expanded[1]:mouth_rect_expanded[3],
                                                    mouth_rect_expanded[0]:mouth_rect_expanded[2]] = 0

        # Draw lips polygon in frame
        frame_with_blackened_mouth_and_lip_polygons = cv2.drawContours(frame_with_blackened_mouth_and_lip_polygons,
                                                                       [mouth_landmarks[:12], mouth_landmarks[12:]], -1, (255, 255, 255))

        return frame_with_blackened_mouth_and_lip_polygons


def expand_rect(rect, scale=None, scale_w=1.5, scale_h=1.5, frame_shape=(256, 256)):
    if scale is not None:
        scale_w = scale
        scale_h = scale
    # Rect: (x, y, x+w, y+h)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    # new_w, new_h
    new_w = int(w * scale_w)
    new_h = int(h * scale_h)
    # new_x
    new_x = int(x - (new_w - w)/2)
    if new_x < 0:
        new_w = new_x + new_w
        new_x = 0
    elif new_x + new_w > (frame_shape[1] - 1):
        new_w = (frame_shape[1] - 1) - new_x
    # new_y
    new_y = int(y - (new_h - h)/2)
    if new_y < 0:
        new_h = new_y + new_h
        new_y = 0
    elif new_y + new_h > (frame_shape[0] - 1):
        new_h = (frame_shape[0] - 1) - new_y
    # Return
    return [new_x, new_y, new_x + new_w, new_y + new_h]


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


def read_landmarks_list_from_txt(path):
    landmarks_list = []
    translate_table = dict((ord(char), None) for char in '[],')
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split(" [")
            video_frame_name = row[0]
            landmarks = row[1:]
            landmarks_list.append([video_frame_name] + [[int(e.translate(translate_table)) for e in l.split(" ")] for l in landmarks])
    return landmarks_list


def watch_video(video_frames):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    for f, frame in enumerate(video_frames):
        ax.imshow(frame)
        ax.set_title(str(f))
        fig.canvas.draw()


def plot_2D_landmarks(image, landmarks, save_or_show='show', fig_name='a.png'):
    frame = np.array(image)
    if frame.max() <= 1.:
        max_value = 1
    else:
        max_value() = 255
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for l, landmark in enumerate(landmarks):
        frame[int(landmark[1]-2):int(landmark[1]+2), int(landmark[0]-2):int(landmark[0]+2)] = max_value
    plt.imshow(frame)
    if save_or_show == 'show':
        plt.show()
        plt.close()
    elif save_or_show == 'save':
        plt.savefig(fig_name)
        plt.close()


def plot_3D_landmarks(image, landmarks, save_or_show='show', fig_name='a.png'):
    #TODO: Make this nice
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(landmarks[60:68, 0], landmarks[60:68, 1], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(landmarks[:, 0]*1.2, landmarks[:,1], landmarks[:, 2], c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(landmarks[:17, 0]*1.2, landmarks[:17,1], landmarks[:17,2], color='blue' )
    ax.plot3D(landmarks[17:22, 0]*1.2, landmarks[17:22, 1], landmarks[17:22, 2], color='blue')
    ax.plot3D(landmarks[22:27, 0]*1.2, landmarks[22:27, 1], landmarks[22:27, 2], color='blue')
    ax.plot3D(landmarks[27:31, 0]*1.2, landmarks[27:31, 1], landmarks[27:31, 2], color='blue')
    ax.plot3D(landmarks[31:36, 0]*1.2, landmarks[31:36, 1], landmarks[31:36, 2], color='blue')
    ax.plot3D(landmarks[36:42, 0]*1.2, landmarks[36:42, 1], landmarks[36:42, 2], color='blue')
    ax.plot3D(landmarks[42:48, 0]*1.2, landmarks[42:48, 1], landmarks[42:48, 2], color='blue')
    ax.plot3D(landmarks[48:60, 0]*1.2, landmarks[48:60, 1], landmarks[48:60, 2], color='blue' )
    ax.plot3D(landmarks[60:, 0]*1.2, landmarks[60:, 1], landmarks[60:, 2], color='blue' )
    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    if save_or_show == 'show':
        plt.show()
        plt.close()
    elif save_or_show == 'save':
        plt.savefig(fig_name)
        plt.close()


def get_video_frames_dir(language, actor, number):
    frames_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'frames', language, actor, actor + '_%04d' % number)
    if not os.path.exists(frames_dir):
        raise ValueError("[ERROR]: frames_dir", frames_dir, "does not exist!")
    else:
        return frames_dir

