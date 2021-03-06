import cv2
import numpy as np
import imageio
import os
import tqdm

from morph_video_with_new_lip_landmarks import *

import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import utils

# video_file_name = '/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4'

def make_combined_face_with_bmp(video_file_name, output_dir_name, align=False):

    # Read video
    cap = cv2.VideoCapture(video_file_name)
    """
    video_reader = imageio.get_reader(video_file_name)
    frames = []
    for frame in tqdm.tqdm(video_reader):
        frames.append(frame)
    """

    if os.path.basename(video_file_name)[2] == '.':
        video_file_name = "CV_" + '_'.join('_'.join(os.path.splitext(os.path.basename(video_file_name))[0].split('.')).split(' '))

    # Read landmarks
    landmarks, frames_with_no_landmarks = read_video_landmarks(video_file_name=video_file_name)

    # MAKE FACE WITH BMP
    for i, (landmarks_in_frame, frame_with_no_landmarks) in tqdm.tqdm(enumerate(zip(landmarks, frames_with_no_landmarks)),
                                                                             total=len(landmarks)):
        if not frame_with_no_landmarks:

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                # Extract face_rect, landmarks
                face_square_expanded_resized, \
                    landmarks_in_face_square_expanded_resized, \
                    face_rect_in_frame, face_original_size = utils.get_square_expand_resize_face_and_modify_landmarks(np.array(frame),
                                                                                                                      landmarks_in_frame,
                                                                                                                      resize_to_shape=(256, 256),
                                                                                                                      face_square_expanded_resized=True)
                # Make black mouth and lips polygons
                mouth_landmarks_in_face = landmarks_in_face_square_expanded_resized[48:68]
                face_square_expanded_resized_with_bmp = utils.make_black_mouth_and_lips_polygons(face_square_expanded_resized,
                                                                                                 mouth_landmarks_in_face,
                                                                                                 align=align)
    
                # Make combined face
                face_combined = np.hstack((face_square_expanded_resized, face_square_expanded_resized_with_bmp))
    
                # Save combined face
                video_base_name = os.path.splitext(os.path.basename(video_file_name))[0]
                image_name = os.path.join(output_dir_name, video_base_name, video_base_name + "_combined_faces_frame_{0:05d}.png".format(i))
                if not os.path.exists(os.path.dirname(image_name)):
                    os.makedirs(os.path.dirname(image_name))
                imageio.imwrite(image_name, face_combined)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='make_combined_face_with_bmp')
    parser.add_argument('video_file_name', type=str, help="target video (.mp4); eg. /home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4")
    parser.add_argument('--output_dir_name', '-o', type=str, help="output_dir_name; eg. /home/voleti.vikram/ANDREW_NG/faces_combined_new/")
    parser.add_argument('--align', '-a', action="store_true")
    
    args = parser.parse_args()

    print(args)

    make_combined_face_with_bmp(args.video_file_name, args.output_dir_name, args.align)
