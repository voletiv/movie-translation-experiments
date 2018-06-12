import cv2
import imageio
# import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

from skimage.transform import resize

from config import *


config = MovieTranslationConfig()


def load_generator(model_path, verbose=False):
    if not os.path.exists(model_path):
        raise ValueError("[ERROR] model path does not exist! Given:", model_path)

    from keras.models import load_model
    if verbose:
        print("Loading model", model_path, "...")

    try:
        model = load_model(model_path)
    except:
        print("Loading", '/'.join(os.path.splitext(model_path)[0].split('/')[:-1] + ['generator_latest.h5']), "instead...")
        latest_model = '/'.join(os.path.splitext(model_path)[0].split('/')[:-1] + ['generator_latest.h5'])
        model = load_model(latest_model)
        model.load_weights(model_path)

    return model


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


def load_face_alignment_object(d='3D', enable_cuda=False, flip_input=False, use_cnn_face_detector=False, verbose=False):
    # Check https://github.com/1adrianb/face-alignment for installation instructions
    if verbose:
        print("Loading FaceAlignment object ...")
    import face_alignment
    if d == '3D':
        return face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                            enable_cuda=enable_cuda,
                                            flip_input=flip_input,
                                            use_cnn_face_detector=use_cnn_face_detector)
    if d == '2D':
        return face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                            enable_cuda=enable_cuda,
                                            flip_input=flip_input,
                                            use_cnn_face_detector=use_cnn_face_detector)


def load_dlib_detector_predictor_facerec(config):
    import dlib
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)
    dlib_facerec = dlib.face_recognition_model_v1(config.FACE_REC_MODEL_PATH)
    return dlib_detector, dlib_predictor, dlib_facerec


def load_landmarks_detectors(load_2D=True, load_3D=True):

    # Load dlib landmarks detector for 2D
    if load_2D:
        dlib_detector, dlib_predictor = utils.load_dlib_detector_and_predictor()
    else:
        dlib_detector, dlib_predictor = None, None

    # Load LS3D-W landmarks detector for 3D
    if load_3D:
        face_alignment_object = utils.load_face_alignment_object(enable_cuda=config.ENABLE_CUDA)
    else:
        face_alignment_object = None

    return dlib_detector, dlib_predictor, face_alignment_object


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


def get_all_landmarks_using_FaceAlignment(frame, face_alignment_object):
    landmarks = face_alignment_object.get_landmarks(frame)
    if landmarks is not None:
        return np.round(landmarks).astype('int')
    else:
        return None


def get_landmarks_using_dlib_detector_and_predictor(frame, detector, predictor):
    import dlib
    # Landmarks Coords: ------> x (cols)
    #                  |
    #                  |
    #                  v
    #                  y
    #                (rows)
    faces = detector(frame, 1)
    if len(faces) > 0:
        # TODO: Use dlib recognition model to verify face
        # Choose first face
        face = faces[0]
        face_exp = expand_rect([face.left(), face.top(), face.right(), face.bottom()], scale=1.5, frame_shape=frame.shape)
        face_exp = dlib.rectangle(face_exp[0], face_exp[1], face_exp[2], face_exp[3])
        # Detect landmarks
        shape = predictor(frame, face_exp)
        landmarks = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
        return np.round(landmarks).astype('int')
    else:
        return None


def shape_to_landmarks(shape):
    return np.round([[shape.part(i).x, shape.part(i).y] for i in range(68)]).astype('int')


def get_all_face_shapes(img, detector, predictor):
    import dlib
    faces = detector(img, 1)
    face_shapes = []
    for face in faces:
        face_exp = expand_rect([face.left(), face.top(), face.right(), face.bottom()], scale=1.5, frame_shape=img.shape)
        face_exp = dlib.rectangle(face_exp[0], face_exp[1], face_exp[2], face_exp[3])
        shape = predictor(img, face_exp)
        # landmarks = np.round([[shape.part(i).x, shape.part(i).y] for i in range(68)]).astype('int')
        face_shapes.append(shape)
    return face_shapes


def get_face_descriptor(img, shape, facerec):
    # facerec = dlib.face_recognition_model_v1(config.FACE_REC_MODEL_PATH)
    return facerec.compute_face_descriptor(img, shape)


def make_rect_shape_square(rect):
    # Rect: (x, y, x+w, y+h)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    # If width > height
    if w > h:
        new_x = x
        new_y = int(y - (w-h)/2)
        new_w = w
        new_h = w
    # Else (height > width)
    else:
        new_x = int(x - (h-w)/2)
        new_y = y
        new_w = h
        new_h = h
    # Return
    return [new_x, new_y, new_x + new_w, new_y + new_h]


def get_square_expand_resize_face_and_modify_landmarks(frame, landmarks, resize_to_shape=(224, 224), face_square_expanded_resized=True):

    # Get face bounding box from landmarks
    # dlib.rectangle = left, top, right, bottom
    if face_square_expanded_resized:
        # face_rect = dlib.rectangle(int(np.min(landmarks[:, 0])), int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 0])), int(np.max(landmarks[:, 1])))
        face_rect = [int(np.min(landmarks[:, 0])), int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 0])), int(np.max(landmarks[:, 1]))]

        # Make face bounding box square to the greater of width and height
        face_rect_square = make_rect_shape_square(face_rect)

        # Expand face bounding box to 1.5x
        face_rect_square_expanded = expand_rect(face_rect_square, scale=1.5, frame_shape=(frame.shape[0], frame.shape[1]))

    else:
        face_rect_square_expanded = [0, 0, frame.shape[1], frame.shape[0]]
    
    # Resize frame[face_bounding_box] to resize_to_shape
    face_square_expanded = frame[face_rect_square_expanded[1]:face_rect_square_expanded[3], face_rect_square_expanded[0]:face_rect_square_expanded[2]]
    face_original_size = face_square_expanded.shape[:2]
    face_square_expanded_resized = np.round(resize(face_square_expanded, resize_to_shape, mode='reflect', preserve_range=True)).astype('uint8')

    # Note the landmarks in the expanded resized face
    # 2D landmarks
    if len(landmarks[0]) == 2:
        landmarks_in_face_square_expanded_resized = np.round([[(x-face_rect_square_expanded[0])/(face_rect_square_expanded[2] - face_rect_square_expanded[0])*resize_to_shape[1],
                                                               (y-face_rect_square_expanded[1])/(face_rect_square_expanded[3] - face_rect_square_expanded[1])*resize_to_shape[0]] for (x, y) in landmarks]).astype('int')
    # 3D landmarks
    elif len(landmarks[0]) == 3:
        landmarks_in_face_square_expanded_resized = np.round([[(x-face_rect_square_expanded[0])/(face_rect_square_expanded[2] - face_rect_square_expanded[0])*resize_to_shape[1],
                                                               (y-face_rect_square_expanded[1])/(face_rect_square_expanded[3] - face_rect_square_expanded[1])*resize_to_shape[0],
                                                               z] for (x, y, z) in landmarks.astype('float')]).astype('int')
                                                               # z/(face_rect_square_expanded[3] - face_rect_square_expanded[1])*224] for (x, y, z) in landmarks.astype('float')]).astype('int')

    return face_square_expanded_resized, landmarks_in_face_square_expanded_resized, face_rect_square_expanded, face_original_size


def align_and_normalize_lm(lm):
    angle = np.arctan((lm[1, 6] - lm[1, 0])/(lm[0, 6] - lm[0, 0] + 1e-8))
    rot_lm = np.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], lm)
    aligned_lm = (rot_lm - rot_lm[:, 0].reshape(2, 1)) / (np.max(rot_lm[0]) - np.min(rot_lm[0]) + 1e-8) * 2 - np.array([[1], [0]])
    aligned_lm[aligned_lm > 1.] = 1.
    aligned_lm[aligned_lm < -1.] = -1.
    return aligned_lm


def align_lm(lm):
    angle = np.arctan((lm[1, 6] - lm[1, 0])/(lm[0, 6] - lm[0, 0] + 1e-8))
    rot_lm = np.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], lm)
    return rot_lm


def make_black_mouth_and_lips_polygons(frame, mouth_landmarks, align=False):

#         # Find mouth bounding box
#         mouth_rect = [int(np.min(mouth_landmarks[:, 0])), int(np.min(mouth_landmarks[:, 1])), int(np.max(mouth_landmarks[:, 0])), int(np.max(mouth_landmarks[:, 1]))]

#         # Expand mouth bounding box
#         mouth_rect_expanded = expand_rect(mouth_rect, scale_w=1.2, scale_h=1.8, frame_shape=(frame.shape[0], frame.shape[1]))

        # Get mouth landmarks centroid
        mouth_centroid = np.mean(mouth_landmarks, axis=0).astype(int)

        # Make mouth_rect as 1/3rd face_width and 1/3rd face_height
        height, width = frame.shape[0], frame.shape[1]
        mouth_rect_expanded = [int(mouth_centroid[0] - width/3/2), int(mouth_centroid[1] - height/3/2),
                               int(mouth_centroid[0] + width/3/2), int(mouth_centroid[1] + height/3/2)]

        # Make new frame for blackened mouth and lip polygons
        frame_with_blackened_mouth_and_lip_polygons = np.array(frame)

        # Blacken (expanded) mouth in frame
        frame_with_blackened_mouth_and_lip_polygons[mouth_rect_expanded[1]:mouth_rect_expanded[3],
                                                    mouth_rect_expanded[0]:mouth_rect_expanded[2]] = 0

        # Align lip landmarks
        if align:
            mouth_landmarks = align_lm(mouth_landmarks)
        
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


def normalize_input_to_generator(list_of_frames):
    return np.array(list_of_frames)/127. - 1.


def unnormalize_output_from_generator(np_array_output_of_generator):
    return np.round((np_array_output_of_generator + 1)/2.*255.).astype('uint8')


def save_new_video_frames_with_target_audio_as_mp4(frames, video_fps, target_audio_file, output_file_name='new_video.mp4', overwrite=False, frame_shape=None, verbose=False):

    # Save mp4 of frames
    if target_audio_file is not None:
        if verbose:
            print("Writing frames as mp4")
        imageio.mimwrite('/tmp/video.mp4', frames, fps=video_fps)
    
        # Convert audio into aac (to integrate into video)
        command = ['ffmpeg', '-loglevel', 'error', '-i', target_audio_file, '-y', '-vn', '-acodec', 'aac', '-strict', '-2', '/tmp/video_audio.aac']
        command_return = subprocess.call(command)
    
        # Combine frames with audio
        output_dir = os.path.dirname(os.path.realpath(output_file_name))
        if not os.path.exists(output_dir):
            print("Making", output_dir)
            os.makedirs(output_dir)

        command = ['ffmpeg', '-loglevel', 'error']

        if overwrite:
            command += ['-y']

        command += ['-i', '/tmp/video.mp4', '-i', '/tmp/video_audio.aac']

        if frame_shape is not None:
            command += ['-s', str(frame_shape[0])+'x'+str(frame_shape[1])]

        command += ['-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main', '-acodec', 'aac', '-strict', '-2',
                    output_file_name]

        if verbose:
            print("Combining new frames with target audio:", command)
        command_return = subprocess.call(command)    # subprocess.call returns 0 on successful run

    else:
        imageio.mimwrite(output_file_name, frames, fps=video_fps)

    # # Save npz
    # if verbose:
    #     print("Saving npz")
    # np.savez("exchanged_dialogues", new_video1=new_video1_frames_generated, new_video2=new_video2_frames_generated)

    # # Save GIF
    # if verbose:
    #     print("Saving gifs")
    # imageio.mimsave(os.path.join("video1.gif"), new_video1_frames_generated)
    # imageio.mimsave(os.path.join("video2.gif"), new_video2_frames_generated)


def read_landmarks(language, actor, number, read_2D_dlib_or_3D=''):
    '''
    language: e.g. 'telugu'
    actor: e.g. 'Mahesh_Babu'
    number: e.g. 3
    read_2D_dlib_or_3D = '' or '2D_dlib' or '3D'
    '''
    if read_2D_dlib_or_3D:
        actor_suffix = '_' + read_2D_dlib_or_3D
    else:
        actor_suffix = ''
    landmarks_file = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'landmarks', language, actor + actor_suffix, actor + '_%04d' % number + "_landmarks.txt")
    if not os.path.exists(landmarks_file):
        raise ValueError("[ERROR] landmarks file does not exist! Given: " + landmarks_file)
    else:
        return read_landmarks_list_from_txt(landmarks_file)


"""
def read_landmarks_list_from_txt(path):
    # Path points to a text file containing rows of [frame_name, [landmarks1], [[0, 0], [1, 1], [2, 2], ...], ...] (len = 1 + num_of_faces, each face landmarks len = 68)
    # landmarks_list is returned as a list of frames - list of faces in each frame - list of landmarks in each face (68)
    landmarks_list = []
    translate_table = dict((ord(char), None) for char in '[],')
    with open(path, "r") as f:
        for line in f:
            frame_landmarks_all = line.strip().split(" [[")[1:]
            # Backward compatibility with previous format of only 1 landmark in frame: [frame_name, [0, 0], [1, 1], ...] (len = 69)
            if frame_landmarks_all == []:
                frame_landmarks_all = [" [".join(line.strip().split(" [")[1:])]
            if len(frame_landmarks_all):
                frame_landmarks = []
                for lm in frame_landmarks_all:
                    person_landmarks = []
                    for l in lm.split(' ['):
                        person_landmarks.append([int(e.translate(translate_table)) for e in l.split(" ")])
                    frame_landmarks.append(person_landmarks)
                landmarks_list.append(frame_landmarks)
    return landmarks_list
"""


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


def write_landmarks_list_as_txt(path, landmarks_list):
    with open(path, "w") as f:
        for frame_landmarks in landmarks_list:
            line = ""
            for landmark in frame_landmarks:
                line += str(landmark) + " "
            line = line[:-1] + "\n"
            f.write(line)


def write_landmarks_list_as_csv(path, landmarks_list):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(landmarks_list)

def watch_video(video_frames):
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
    # ONLY plt.imshow!! Need to execute plt.show() separately!
    frame = np.array(image)
    if frame.max() <= 1.:
        max_value = 1
    else:
        max_value = 255
    face_width = landmarks[:, 0].max() - landmarks[:, 0].min()
    lm_width = np.ceil(face_width / 60)
    for l, landmark in enumerate(landmarks):
        frame[int(landmark[1]-lm_width):int(landmark[1]+lm_width), int(landmark[0]-lm_width):int(landmark[0]+lm_width)] = max_value
    if save_or_show == 'show':
        plt.imshow(frame)
    elif save_or_show == 'save':
        imageio.imwrite(fig_name, frame)


def plot_3D_landmarks(image, landmarks, save_or_show='show', fig_name='a.png'):
    import matplotlib.pyplot as plt
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

