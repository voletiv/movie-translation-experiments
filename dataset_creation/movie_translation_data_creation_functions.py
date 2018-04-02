import csv
import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import tqdm

from skimage.transform import resize

from movie_translation_data_creation_params import *

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


def extract_video_clips(language, actor, metadata, youtube_videos_dir=os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "youtube_videos"), verbose=False):

    # Make video_clips_dir
    video_clips_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "videos", language, actor)
    if not os.path.exists(video_clips_dir):
        if verbose:
            print("Making dir", video_clips_dir)
        os.makedirs(video_clips_dir)

    # Read each column of metadata
    # output_video_file_name | youtube_URL | start_time | duration
    output_video_file_names = []
    youtube_URLs = []
    start_times = []
    durations = []
    for line in metadata:
        output_video_file_names.append(line[0])
        youtube_URLs.append(line[1])
        start_times.append(line[2])
        durations.append(line[3])

    # Extract clips
    for output_video_file_name, youtube_URL, start_time, duration in tqdm.tqdm(zip(output_video_file_names, youtube_URLs, start_times, durations), total=len(durations)):
        output_video = os.path.join(video_clips_dir, output_video_file_name)
        video1 = os.path.join(youtube_videos_dir, language, youtube_URL + '.mp4')
        # ffmpeg -ss 00:08:31 -i LS6XiINMc2s.mp4 -t 00:00:01.5 -y -vcodec libx264 -preset ultrafast -profile:v main -acodec aac -strict -2 newStream1.mp4
        command = ['ffmpeg', '-loglevel', 'warning', '-ss', start_time, '-i', video1, '-t', duration, '-y',
                   '-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main', '-acodec', 'aac', '-strict', '-2', output_video]
        if verbose:
            print(" ".join(command))
        subprocess.call(command)


def extract_face_frames_and_landmarks_from_video(video_file, using_dlib_or_face_alignment,
                                                 dlib_detector=None, dlib_predictor=None, face_alignment_object=None,
                                                 crop_expanded_face_square=True,
                                                 save_with_blackened_mouths_and_polygons=True,
                                                 save_gif=False,
                                                 save_landmarks_as_txt=True, save_landmarks_as_csv=False,
                                                 verbose=False):
    '''
    Extract face frames using landmarks, and save in MOVIE_TRANSLATION_DATASET_DIR/frames/language/actor/video_file
    video_file: .mp4 file from which to extract frames, and possibly landmarks, e.g. '/home/voletiv/Datasets/MOVIE_TRANSLATION/videos/telugu/Mahesh_Babu/Mahesh_Babu_0000.mp4'
    using_dlib_or_face_alignment: either 'dlib' or 'face_alignment';
                                  in case of dlib, make sure to also import 'dlib_detector' and 'dlib_predictor', made using 'load_dlib_detector_and_predictor' function 
                                  in case of face_alignment, make sure to also import 'face_alignment_object' using load_face_alignment_object()
    [optional] Also save all face frames combined with frames with blackened mouth and lip polygons
    [optional] Save all face frames as gif
    [optional] Save landmarks in MOVIE_TRANSLATION_DATASET_DIR/landmarks/language/actor
    '''

    if using_dlib_or_face_alignment == 'dlib':
        if dlib_detector is None or dlib_predictor is None:
            print("\n\n[ERROR] Please provide dlib_detector and dlib_predictor! (Since you have chosen the option of 'dlib' in 'using_dlib_or_face_alignment')\n\n")
            return

    elif using_dlib_or_face_alignment == 'face_alignment':
        if face_alignment_object is None:
            print("\n\n[ERROR] Please provide face_alignment_object! (Since you have chosen the option of 'face_alignment' in 'using_dlib_or_face_alignment')\n\n")
            return

    video_file_split = video_file.split("/")
    video_file_name = os.path.splitext(video_file_split[-1])[0]
    actor = video_file_split[-2]
    language = video_file_split[-3]
    video_frames_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "frames", language, actor, video_file_name)

    # Make video_frames_dir
    if not os.path.exists(video_frames_dir):
        os.makedirs(video_frames_dir)

    # Make video_frames_combined_dir
    if save_with_blackened_mouths_and_polygons:
        video_frames_combined_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "frames_combined", language, actor, video_file_name)
        if not os.path.exists(video_frames_combined_dir):
            os.makedirs(video_frames_combined_dir)

    # Make landmarks_dir
    if save_landmarks_as_txt or save_landmarks_as_csv:
        landmarks_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "landmarks", language, actor)
        if not os.path.exists(landmarks_dir):
            os.makedirs(landmarks_dir)

    # Read video
    video_frames = imageio.get_reader(video_file)

    if save_gif:
        faces_list = []

    if save_landmarks_as_txt or save_landmarks_as_csv:
        landmarks_list = []

    # For each frame in the video
    for frame_number, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):
        video_frame_name = video_file_name + "_frame_{0:03d}.png".format(frame_number)

        # Get landmarks
        if using_dlib_or_face_alignment == 'dlib':
            landmarks = get_landmarks_using_dlib_detector_and_predictor(frame, dlib_detector, dlib_predictor)

        elif using_dlib_or_face_alignment == 'face_alignment':
            landmarks = get_landmarks_using_FaceAlignment(frame, face_alignment_object)

        # If landmarks are found
        if landmarks is not None:

            # Crop 1.5x face, resize to 224x224, note new landmark locations
            face_square_expanded_resized, landmarks_in_face_square_expanded_resized = square_expand_resize_face_and_modify_landmarks(frame,
                                                                                                                                     landmarks,
                                                                                                                                     crop_expanded_face_square)

            if save_gif:
                faces_list.append(face_square_expanded_resized)

            if save_landmarks_as_txt or save_landmarks_as_csv:
                landmarks_list.append([video_frame_name] + [list(l) for l in landmarks_in_face_square_expanded_resized])

            # Write face image
            cv2.imwrite(os.path.join(video_frames_dir, video_frame_name), cv2.cvtColor(face_square_expanded_resized, cv2.COLOR_RGB2BGR))

            if save_with_blackened_mouths_and_polygons:

                # Make new frame with blackened mouth
                mouth_landmarks = landmarks_in_face_square_expanded_resized[48:68, :2]
                face_with_blackened_mouth_and_mouth_polygon = make_black_mouth_and_lips_polygons(face_square_expanded_resized, mouth_landmarks)

                # Write combined frame+frame_with_blacked_mouth_and_polygon image
                face_combined = np.hstack((face_square_expanded_resized, face_with_blackened_mouth_and_mouth_polygon))
                video_frame_combined_name = video_file_name + "_frame_combined_{0:03d}.png".format(frame_number)
                cv2.imwrite(os.path.join(video_frames_combined_dir, video_frame_combined_name), cv2.cvtColor(face_combined, cv2.COLOR_RGB2BGR))

        # If landmarks are not found
        else:
            if save_landmarks_as_txt or save_landmarks_as_csv:
                landmarks_list.append([video_frame_name] + [])

    # Save gif
    if save_gif:
        imageio.mimsave(os.path.join(video_frames_dir, video_file_name + ".gif"), faces_list)

    # Save landmarks
    # txt is smaller than csv
    if save_landmarks_as_txt:
        write_landmarks_list_as_txt(os.path.join(landmarks_dir, video_file_name + "_landmarks.txt"), landmarks_list)
    if save_landmarks_as_csv:
        write_landmarks_list_as_csv(os.path.join(landmarks_dir, video_file_name + "_landmarks.csv"), landmarks_list)


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


def square_expand_resize_face_and_modify_landmarks(frame, landmarks, face_square_expanded_resized=True):

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
    
    # Resize frame[face_bounding_box] to 224x224
    face_square_expanded = frame[face_rect_square_expanded[1]:face_rect_square_expanded[3], face_rect_square_expanded[0]:face_rect_square_expanded[2]]
    face_square_expanded_resized = np.round(resize(face_square_expanded, (224, 224), preserve_range=True)).astype('uint8')

    # Note the landmarks in the expanded resized face
    # 2D landmarks
    if len(landmarks[0]) == 2:
        landmarks_in_face_square_expanded_resized = np.round([[(x-face_rect_square_expanded[0])/(face_rect_square_expanded[2] - face_rect_square_expanded[0])*224,
                                                               (y-face_rect_square_expanded[1])/(face_rect_square_expanded[3] - face_rect_square_expanded[1])*224] for (x, y) in landmarks]).astype('int')
    # 3D landmarks
    elif len(landmarks[0]) == 3:
        landmarks_in_face_square_expanded_resized = np.round([[(x-face_rect_square_expanded[0])/(face_rect_square_expanded[2] - face_rect_square_expanded[0])*224,
                                                               (y-face_rect_square_expanded[1])/(face_rect_square_expanded[3] - face_rect_square_expanded[1])*224,
                                                               z] for (x, y, z) in landmarks]).astype('int')
                                                               # z/(face_rect_square_expanded[3] - face_rect_square_expanded[1])*224] for (x, y, z) in landmarks])

    return face_square_expanded_resized, landmarks_in_face_square_expanded_resized


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


def write_landmarks_list_as_csv(path, landmarks_list):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(landmarks_list)


def write_landmarks_list_as_txt(path, landmarks_list):
    with open(path, "w") as f:
        for landmarks_of_file in landmarks_list:
            line = ""
            for landmark in landmarks_of_file:
                line += str(landmark) + " "
            line = line[:-1] + "\n"
            f.write(line)


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


def plot_landmarks(frame, landmarks):
    frame = np.array(frame)
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    plt.imshow(frame)
    plt.show()


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


def correct_video_numbers_in_metadata(d, actor, write=False, txt_file_path=None):
    for i, l in enumerate(d):
        l[0] = actor + '_{0:04d}.mp4'.format(i)
    if write:
        if txt_file_path is None:
            txt_file_path = os.path.join(os.getcwd(), actor + '_.txt')
        print("Writing corrected metadata in", )
        with open(txt_file_path, 'w') as f:
            for l in d:
                f.write(" ".join(l) + "\n")
    return d


def write_combined_frames_with_blackened_mouths_and_mouth_polygons(video_name):
    # Read landmarks
    landmarks_file = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'landmarks', language, actor, video_name + "_landmarks.txt")
    video_landmarks = read_landmarks_list_from_txt(landmarks_file)
    video_frames_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'frames', language, actor, video_name)
    # Folders
    video_frames_and_black_mouths_combined_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'frames_combined', language, actor, video_name)
    if not os.path.exists(video_frames_and_black_mouths_combined_dir):
        os.makedirs(video_frames_and_black_mouths_combined_dir)
    # For each frame
    for frame_number, frame_name_and_landmarks in enumerate(video_landmarks):
        frame_name = frame_name_and_landmarks[0]
        frame_landmarks = frame_name_and_landmarks[1:]
        if len(frame_name_and_landmarks) == 69:
            frame = cv2.cvtColor(cv2.imread(os.path.join(video_frames_dir, frame_name)), cv2.COLOR_BGR2RGB)
            mouth_landmarks = np.array(frame_landmarks[48:68])[:, :2]
            frame_with_blackened_mouth_and_lip_polygons = make_black_mouth_and_lips_polygons(frame, mouth_landmarks)
            # Write image
            frame_combined = np.hstack((frame, frame_with_blackened_mouth_and_mouth_polygon))
            cv2.imwrite(os.path.join(video_frames_and_black_mouths_combined_dir, video_name + "_frame_combined_{0:03d}.png".format(frame_number)), cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR))


def read_log_and_plot_graphs(log_txt_path):
    log_lines = []
    with open(log_txt_path, 'r') as f:
        for line in f:
            log_lines.append(line)
    D_log_losses = []
    G_tot_losses = []
    G_l1_losses = []
    G_log_losses = []
    for line in log_lines:
        if 'D logloss' in line:
            line_split = line.split()
            D_log_losses.append(float(line_split[8]))
            G_tot_losses.append(float(line_split[12]))
            G_l1_losses.append(float(line_split[16]))
            G_log_losses.append(float(line_split[20][:6]))
    plt.figure()
    plt.subplot(121)
    plt.plot(np.arange(len(D_log_losses)), D_log_losses)
    plt.xlabel("Epochs")
    plt.title("Discriminator loss")
    plt.subplot(122)
    plt.plot(np.arange(len(G_tot_losses)), G_tot_losses, linewidth=2, label='G_total_loss')
    plt.plot(np.arange(len(G_l1_losses)), G_l1_losses, label='G_L1_loss')
    plt.plot(np.arange(len(G_log_losses)), G_log_losses, label='G_log_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.title("Generator loss")
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
        frame[int(lip_landmark[1]-2):int(lip_landmark[1]+2), int(lip_landmark[0]-2):int(lip_landmark[0]+2)] = 1
        ax.imshow(frame)
        if video:
            ax.set_title(str(l))
            fig.canvas.draw()
    if not video:
        plt.show()


def plot_2D_landmarks(image, landmarks, save_or_show='show', fig_name='a.png'):
    frame = np.array(image)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for l, landmark in enumerate(landmarks):
        frame[int(landmark[1]-2):int(landmark[1]+2), int(landmark[0]-2):int(landmark[0]+2)] = 0
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
