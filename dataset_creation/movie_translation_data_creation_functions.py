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
                                                 dlib_detector=None, dlib_predictor=None,
                                                 face_alignment_3D_object=None, face_alignment_2D_object=None,
                                                 crop_expanded_face_square=True, resize_to_shape=(256, 256),
                                                 save_with_blackened_mouths_and_polygons=True,
                                                 save_landmarks_as_txt=True, save_landmarks_as_csv=False,
                                                 skip_frames=0, check_for_face_every_nth_frame=10,
                                                 output_dir=config.MOVIE_TRANSLATION_DATASET_DIR,
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
        if face_alignment_3D_object is None or face_alignment_2D_object is None:
            print("\n\n[ERROR] Please provide face_alignment_object! (Since you have chosen the option of 'face_alignment' in 'using_dlib_or_face_alignment')\n\n")
            return

    video_file_split = video_file.split("/")
    actor = video_file_split[-2]
    language = video_file_split[-3]
    video_file_name = os.path.splitext('_'.join(video_file_split[video_file_split.index(actor):]))[0].replace(" ", "_")
    video_frames_dir = os.path.join(output_dir, "frames", language, actor, video_file_name)

    # Make video_frames_dir
    if not os.path.exists(video_frames_dir):
        os.makedirs(video_frames_dir)

    # Make video_faces_combined_dir
    if save_with_blackened_mouths_and_polygons:
        video_faces_combined_dir = os.path.join(output_dir, "faces_combined", language, actor, video_file_name)
        if not os.path.exists(video_faces_combined_dir):
            os.makedirs(video_faces_combined_dir)
        video_faces_combined_3D_dir = os.path.join(output_dir, "faces_combined_using_3D_landmarks", language, actor, video_file_name)
        if not os.path.exists(video_faces_combined_3D_dir):
            os.makedirs(video_faces_combined_3D_dir)
        video_faces_combined_2D_dir = os.path.join(output_dir, "faces_combined_using_2D_landmarks", language, actor, video_file_name)
        if not os.path.exists(video_faces_combined_2D_dir):
            os.makedirs(video_faces_combined_2D_dir)

    # Make landmarks_in_frames dir
    if save_landmarks_as_txt or save_landmarks_as_csv:
        if using_dlib_or_face_alignment == 'dlib':
            landmarks_in_frames_dir = os.path.join(output_dir, "landmarks_in_frames", language, actor)
            if not os.path.exists(landmarks_in_frames_dir):
                os.makedirs(landmarks_in_frames_dir)
            landmarks_3D_in_frames_dir = None
            landmarks_2D_in_frames_dir = None
        elif using_dlib_or_face_alignment == 'face_alignment':
            landmarks_in_frames_dir = None
            landmarks_3D_in_frames_dir = os.path.join(output_dir, "landmarks_3D_in_frames", language, actor)
            if not os.path.exists(landmarks_3D_in_frames_dir):
                os.makedirs(landmarks_3D_in_frames_dir)
            landmarks_2D_in_frames_dir = os.path.join(output_dir, "landmarks_2D_in_frames", language, actor)
            if not os.path.exists(landmarks_2D_in_frames_dir):
                os.makedirs(landmarks_2D_in_frames_dir)

    # Make landmarks_in_faces dir
    if save_landmarks_as_txt or save_landmarks_as_csv:
        if using_dlib_or_face_alignment == 'dlib':
            landmarks_in_faces_dir = os.path.join(output_dir, "landmarks_in_faces", language, actor)
            if not os.path.exists(landmarks_in_faces_dir):
                os.makedirs(landmarks_in_faces_dir)
            landmarks_3D_in_faces_dir = None
            landmarks_2D_in_faces_dir = None
        elif using_dlib_or_face_alignment == 'face_alignment': 
            landmarks_in_faces_dir = None
            landmarks_3D_in_faces_dir = os.path.join(output_dir, "landmarks_3D_in_faces", language, actor)
            if not os.path.exists(landmarks_3D_in_faces_dir):
                os.makedirs(landmarks_3D_in_faces_dir)
            landmarks_2D_in_faces_dir = os.path.join(output_dir, "landmarks_2D_in_faces", language, actor)
            if not os.path.exists(landmarks_2D_in_faces_dir):
                os.makedirs(landmarks_2D_in_faces_dir)

    # Read video
    video_frames = imageio.get_reader(video_file)

    # if save_landmarks_as_txt or save_landmarks_as_csv:
    landmarks_in_frames_list = []
    landmarks_3D_in_frames_list = []
    landmarks_2D_in_frames_list = []
    landmarks_in_faces_list = []
    landmarks_3D_in_faces_list = []
    landmarks_2D_in_faces_list = []

    try:

        # Check for face only every 10th frame
        # Meanwhile, save the 10 frames of a batch in an array
        # If 10th frame has face, or prev_batch had face, check for facial landmarks, etc.
        # in all frames of batch
        batch_frame_numbers = []
        batch_frames = []
        detect_face_shapes = False

        # For each frame in the video
        for frame_number, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):

            if frame_number < skip_frames:
                continue

            # Till frame_number+1 % 10 == 0, only save the frames
            if (frame_number + 1) % check_for_face_every_nth_frame != 0:
                batch_frame_numbers.append(frame_number)
                batch_frames.append(frame)
                continue

            # At the 10th frame
            batch_frame_numbers.append(frame_number)
            batch_frames.append(frame)

            # Check if 10th frame has faces
            # Get landmarks
            if using_dlib_or_face_alignment == 'dlib':
                landmarks = utils.get_landmarks_using_dlib_detector_and_predictor(frame, dlib_detector, dlib_predictor)

            elif using_dlib_or_face_alignment == 'face_alignment':
                landmarks = utils.get_landmarks_using_FaceAlignment(frame, face_alignment_2D_object)

            # If person's face is not present, and was not present in previous batch - reset and continue
            if landmarks is None and not detect_face_shapes:
                if verbose:
                    print("\nNo person face detected in 10th frame of batch\n")
                detect_face_shapes = False

            # Else if person's face is not present, but was present in previous batch - detect faces for all frames in batch, and deactivate detect_face_shapes
            elif landmarks is None and detect_face_shapes:
                if verbose:
                    print("\nNo person face detected in 10th frame of batch, but was in prev_batch, so running\n")
                for batch_frame_number, batch_frame in zip(batch_frame_numbers, batch_frames):

                    landmarks_in_frames_list, landmarks_3D_in_frames_list, landmarks_2D_in_frames_list, \
                        landmarks_in_faces_list, landmarks_3D_in_faces_list, landmarks_2D_in_faces_list = get_landmarks_and_save(batch_frame_number, batch_frame, video_file_name,
                            using_dlib_or_face_alignment, dlib_detector, dlib_predictor, face_alignment_3D_object, face_alignment_2D_object,
                            crop_expanded_face_square, resize_to_shape, save_with_blackened_mouths_and_polygons,
                            landmarks_in_frames_list, landmarks_3D_in_frames_list, landmarks_2D_in_frames_list,
                            landmarks_in_faces_list, landmarks_3D_in_faces_list, landmarks_2D_in_faces_list,
                            video_faces_combined_dir, video_faces_combined_3D_dir, video_faces_combined_2D_dir)

                detect_face_shapes = False

            # Else if person's face is present, detect faces for all frames in batch, and activate detect_face_shapes
            elif landmarks is not None:
                if verbose:
                    print("\nPerson face detected in 10th frame of batch, running\n")
                for batch_frame_number, batch_frame in zip(batch_frame_numbers, batch_frames):

                    landmarks_in_frames_list, landmarks_3D_in_frames_list, landmarks_2D_in_frames_list, \
                        landmarks_in_faces_list, landmarks_3D_in_faces_list, landmarks_2D_in_faces_list = get_landmarks_and_save(batch_frame_number, batch_frame, video_file_name,
                            using_dlib_or_face_alignment, dlib_detector, dlib_predictor, face_alignment_3D_object, face_alignment_2D_object,
                            crop_expanded_face_square, resize_to_shape, save_with_blackened_mouths_and_polygons,
                            landmarks_in_frames_list, landmarks_3D_in_frames_list, landmarks_2D_in_frames_list,
                            landmarks_in_faces_list, landmarks_3D_in_faces_list, landmarks_2D_in_faces_list,
                            video_faces_combined_dir, video_faces_combined_3D_dir, video_faces_combined_2D_dir)

                detect_face_shapes = True

            # Reset batch_frame_numbers and batch_frames
            batch_frame_numbers = []
            batch_frames = []

    except KeyboardInterrupt:
        # Clean exit by saving landmarks
        print("\n\nCtrl+C was pressed!\n\n")

    # Save gif
    # if save_gif:
    #     imageio.mimsave(os.path.join(video_frames_dir, video_file_name + ".gif"), faces_list)

    # Save landmarks
    # txt is smaller than csv
    if save_landmarks_as_txt:
        if using_dlib_or_face_alignment == 'dlib':
            utils.write_landmarks_list_as_txt(os.path.join(landmarks_in_frames_dir, video_file_name + "_landmarks_in_frames.txt"), landmarks_in_frames_list)
            utils.write_landmarks_list_as_txt(os.path.join(landmarks_in_faces_dir, video_file_name + "_landmarks_in_faces.txt"), landmarks_in_faces_list)
        elif using_dlib_or_face_alignment == 'face_alignment':
            utils.write_landmarks_list_as_txt(os.path.join(landmarks_3D_in_frames_dir, video_file_name + "_landmarks_3D_in_frames.txt"), landmarks_3D_in_frames_list)
            utils.write_landmarks_list_as_txt(os.path.join(landmarks_2D_in_frames_dir, video_file_name + "_landmarks_2D_in_frames.txt"), landmarks_2D_in_frames_list)
            utils.write_landmarks_list_as_txt(os.path.join(landmarks_3D_in_faces_dir, video_file_name + "_landmarks_3D_in_faces.txt"), landmarks_3D_in_faces_list)
            utils.write_landmarks_list_as_txt(os.path.join(landmarks_2D_in_faces_dir, video_file_name + "_landmarks_2D_in_faces.txt"), landmarks_2D_in_faces_list)
    if save_landmarks_as_csv:
        utils.write_landmarks_list_as_csv(os.path.join(landmarks_in_frames_dir, video_file_name + "_landmarks_in_frames.csv"), landmarks_in_frames_list)
        utils.write_landmarks_list_as_csv(os.path.join(landmarks_in_faces_dir, video_file_name + "_landmarks_in_faces.csv"), landmarks_in_faces_list)


def get_landmarks_and_save(frame_number, frame, video_file_name,
                           using_dlib_or_face_alignment, dlib_detector, dlib_predictor, face_alignment_3D_object, face_alignment_2D_object,
                           crop_expanded_face_square, resize_to_shape, save_with_blackened_mouths_and_polygons,
                           landmarks_in_frames_list, landmarks_3D_in_frames_list, landmarks_2D_in_frames_list,
                           landmarks_in_faces_list, landmarks_3D_in_faces_list, landmarks_2D_in_faces_list,
                           video_faces_combined_dir, video_faces_combined_3D_dir, video_faces_combined_2D_dir):

            video_frame_name = video_file_name + "_frame_{0:05d}.png".format(frame_number)
            # cv2.imwrite(os.path.join(video_frames_dir, video_frame_name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Get landmarks
            if using_dlib_or_face_alignment == 'dlib':
                landmarks = utils.get_landmarks_using_dlib_detector_and_predictor(frame, dlib_detector, dlib_predictor)
                landmarks_3D = None
                landmarks_2D = None

            elif using_dlib_or_face_alignment == 'face_alignment':
                landmarks = None
                landmarks_3D = utils.get_landmarks_using_FaceAlignment(frame, face_alignment_3D_object)
                landmarks_2D = utils.get_landmarks_using_FaceAlignment(frame, face_alignment_2D_object)

            # If landmarks are found

            if landmarks is not None:                
                # Save landmarks in frames
                landmarks_in_frames_list.append([video_frame_name] + [list(l) for l in landmarks])
                landmarks_in_faces_list = save_landmarks_and_faces_combined(frame, landmarks, resize_to_shape, crop_expanded_face_square,
                                                                            save_with_blackened_mouths_and_polygons,
                                                                            landmarks_in_faces_list,
                                                                            video_file_name, video_frame_name, video_faces_combined_dir)

            if landmarks_3D is not None:
                # Save landmarks in frames
                landmarks_3D_in_frames_list.append([video_frame_name] + [list(l) for l in landmarks_3D])
                landmarks_3D_in_faces_list = save_landmarks_and_faces_combined(frame, landmarks_3D, resize_to_shape, crop_expanded_face_square,
                                                                               save_with_blackened_mouths_and_polygons,
                                                                               landmarks_3D_in_faces_list,
                                                                               video_file_name, video_frame_name, video_faces_combined_3D_dir)

            if landmarks_2D is not None:
                # Save landmarks in frames
                landmarks_2D_in_frames_list.append([video_frame_name] + [list(l) for l in landmarks_2D])
                landmarks_2D_in_faces_list = save_landmarks_and_faces_combined(frame, landmarks_2D, resize_to_shape, crop_expanded_face_square,
                                                                               save_with_blackened_mouths_and_polygons,
                                                                               landmarks_2D_in_faces_list,
                                                                               video_file_name, video_frame_name, video_faces_combined_2D_dir)

            # If landmarks are not found
            # else:
            #     if save_landmarks_as_txt or save_landmarks_as_csv:
            #         landmarks_in_frames_list.append([video_frame_name] + [])
            #         landmarks_in_faces_list.append([video_frame_name] + [])

            return landmarks_in_frames_list, landmarks_3D_in_frames_list, landmarks_2D_in_frames_list, \
                landmarks_in_faces_list, landmarks_3D_in_faces_list, landmarks_2D_in_faces_list


def save_landmarks_and_faces_combined(frame, landmarks, resize_to_shape, crop_expanded_face_square, save_with_blackened_mouths_and_polygons,\
                                      landmarks_in_faces_list, video_file_name, video_frame_name, video_faces_combined_dir):

                # Crop 1.5x face, resize to 224x224, note new landmark locations
                face_square_expanded_resized, \
                    landmarks_in_face_square_expanded_resized, _, _ = utils.get_square_expand_resize_face_and_modify_landmarks(frame,
                                                                                                                               landmarks,
                                                                                                                               resize_to_shape,
                                                                                                                               crop_expanded_face_square)

                # Save landmarks in faces
                landmarks_in_faces_list.append([video_frame_name] + [list(l) for l in landmarks_in_face_square_expanded_resized])

                # Write face image
                # if not save_with_blackened_mouths_and_polygons:
                #     cv2.imwrite(os.path.join(video_faces_dir, video_frame_name), cv2.cvtColor(face_square_expanded_resized, cv2.COLOR_RGB2BGR))

                if save_with_blackened_mouths_and_polygons:

                    # Make new frame with blackened mouth
                    mouth_landmarks = landmarks_in_face_square_expanded_resized[48:68, :2]
                    face_with_blackened_mouth_and_mouth_polygon = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized, mouth_landmarks)

                    # Write combined frame+frame_with_blacked_mouth_and_polygon image
                    faces_combined = np.hstack((face_square_expanded_resized, face_with_blackened_mouth_and_mouth_polygon))
                    video_faces_combined_name = video_file_name + "_faces_combined_{0:05d}.png".format(frame_number)
                    cv2.imwrite(os.path.join(video_faces_combined_dir, video_faces_combined_name), cv2.cvtColor(faces_combined, cv2.COLOR_RGB2BGR))

                return landmarks_in_faces_list


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
    # landmarks_file = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'landmarks', language, actor, video_name + "_landmarks.txt")
    # video_landmarks = read_landmarks_list_from_txt(landmarks_file)
    video_landmarks = read_landmarks(language, actor, number, read_2D_dlib_or_3D='2D_dlib')
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
            frame_with_blackened_mouth_and_lip_polygons = utils.make_black_mouth_and_lips_polygons(frame, mouth_landmarks)
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

