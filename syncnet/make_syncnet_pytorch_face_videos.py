import argparse
import imageio
import numpy as np
import os
import skimage.transform
import tqdm
import sys

sys.path.append('../andrew_ng/')
from morph_video_with_new_lip_landmarks import read_video_landmarks

sys.path.append('../')
import utils

SYNCNET_IMAGE_SHAPE = (224, 224)
SYNCNET_VIDEO_FPS = 25


def convert_video_fps(video_file_name, required_fps=25, converted_video_file_name='/tmp/video.mp4', frame_shape=SYNCNET_IMAGE_SHAPE):
    # cmd = "ffmpeg -i {} -r {} -s {} -y {}".format(video_file_name, str(int(required_fps)), str(frame_shape[0])+'x'+str(frame_shape[1]), converted_video_file_name)
    cmd = "ffmpeg -loglevel error -i {} -r {} -y {}".format(video_file_name, str(int(required_fps)), converted_video_file_name)
    os.system(cmd)


def make_video_for_syncnet_pytorch(video_file_name, verbose=False):

    # Video
    video_frames = imageio.get_reader(video_file_name)
    video_fps = video_frames.get_meta_data()['fps']
    if verbose:
        print("Read video", video_file_name)
        print("FPS", video_fps)

    # Landmarks
    landmarks_in_frames, frames_with_no_landmarks = read_video_landmarks(video_file_name=video_file_name, video_fps=video_fps, verbose=verbose)

    faces = []

    # Make video of faces for syncnet
    for frame, landmarks_in_frame, no_face_in_frame in tqdm.tqdm(zip(video_frames, landmarks_in_frames, frames_with_no_landmarks),
                                                                 total=len(video_frames)):

        # face_rect
        face_rect = utils.make_rect_shape_square([np.min(landmarks_in_frame[:, 0]), np.min(landmarks_in_frame[:, 1]),
                                                  np.max(landmarks_in_frame[:, 0]), np.max(landmarks_in_frame[:, 1])])
        # Expand face_rect
        face_rect_exp = utils.expand_rect(face_rect, scale=1.5, frame_shape=(frame.shape[0], frame.shape[1]))

        # Extract face
        face = frame[face_rect_exp[1]:face_rect_exp[3], face_rect_exp[0]:face_rect_exp[2]]
        face = np.round(skimage.transform.resize(face, SYNCNET_IMAGE_SHAPE) * 255.).astype('uint8')
        faces.append(face)
    
    # Convert video's audio to .wav file
    audio_file_name = '/tmp/audio.wav'
    command = "ffmpeg -y -loglevel error -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(video_file_name, audio_file_name)
    os.system(command)

    # Save video
    output_file_name = os.path.splitext(video_file_name)[0] + '_faces' + os.path.splitext(video_file_name)[-1]
    print("Saving faces video as", output_file_name)
    utils.save_new_video_frames_with_target_audio_as_mp4(faces, video_fps, audio_file_name,
                                                         output_file_name=output_file_name, verbose=verbose)

    if video_fps != SYNCNET_VIDEO_FPS:
        print("Converting fps from", video_fps, "to", SYNCNET_VIDEO_FPS)
        output_file_name_new_fps = os.path.splitext(video_file_name)[0] + '_faces_new_fps' + os.path.splitext(video_file_name)[-1]
        convert_video_fps(output_file_name, required_fps=SYNCNET_VIDEO_FPS, converted_video_file_name=output_file_name_new_fps)
        os.rename(output_file_name_new_fps, output_file_name)


# TO DELAY VIDEO by 3.84 seconds
# ffmpeg -i "movie.mp4" -itsoffset 3.84 -i "movie.mp4" -map 1:v -map 0:a -c copy "movie-video-delayed.mp4"

# TO DELAY AUDIO by 3.84 seconds
# ffmpeg -i "movie.mp4" -itsoffset 3.84 -i "movie.mp4" -map 0:v -map 1:a -c copy "movie-audio-delayed.mp4"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make ANDREW_NG videos of faces from landmarks')
    parser.add_argument('video_file_name', type=str, help="input video (.mp4); eg. /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4")
    parser.add_argument('--output_video_name', '-o', type=str, default=None, help="Name of output video; def: <input_video_name>_faces.mp4")
    parser.add_argument('--verbose', '-v', action="store_true")
    args = parser.parse_args()

    try:
        make_video_for_syncnet_pytorch(args.video_file_name, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed!\n\n")

