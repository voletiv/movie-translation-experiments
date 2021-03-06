import argparse
import imageio
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from scipy.io import loadmat

import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import utils


def extract_lip_landmarks_from_mat(lip_landmarks_mat_file):
    mat = loadmat(lip_landmarks_mat_file)
    lip_landmarks = mat['y_pred']
    return lip_landmarks


def make_video_of_lip_landmarks(lip_landmarks_mat_file, video_fps=25, prepend_time_in_ms=200, audio_file_name=None, output_file_name='lip_landmarks_video.mp4', ffmpeg_overwrite=False, verbose=False):

    # Extract lip landmarks from mat
    if verbose:
        print("Extracting lip landmarks from mat file")
    lip_landmarks_in_frames = extract_lip_landmarks_from_mat(lip_landmarks_mat_file)

    # Calculate xlim and ylim
    xlim = (np.min(lip_landmarks_in_frames[:, :, 0]) - 0.01, np.max(lip_landmarks_in_frames[:, :, 0]) + 0.01)
    ylim = (np.min(lip_landmarks_in_frames[:, :, 1]) - 0.01, np.max(lip_landmarks_in_frames[:, :, 1]) + 0.01)
    if verbose:
        print("xlim", xlim, "; ylim", ylim)

    frames = []

    tmp_image_name = '/tmp/frame_%s.png' % os.path.basename(lip_landmarks_mat_file)[:-4]

    for lip_landmarks_in_frame in tqdm.tqdm(lip_landmarks_in_frames):
        plt.scatter(lip_landmarks_in_frame[:, 0], -lip_landmarks_in_frame[:, 1])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(tmp_image_name, bbox_inches='tight')
        plt.close()
        frames.append(imageio.imread(tmp_image_name))

    if verbose:
        print("frames", len(frames), frames[0].shape)
        print("Saving frames as video")

    number_of_frames_to_prepend = (video_fps * prepend_time_in_ms / 1000)
    frames = np.concatenate((np.repeat(np.expand_dims(frames[0], axis=0), number_of_frames_to_prepend, axis=0), frames), axis=0)

    utils.save_new_video_frames_with_target_audio_as_mp4(frames, video_fps, audio_file_name, output_file_name=output_file_name, overwrite=ffmpeg_overwrite, verbose=verbose)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make video of lip landmarks')
    parser.add_argument('lip_landmarks_mat_file', type=str, help="lip_landmarks_mat_file (.mat); eg. CV_05_C4W1L05_000001_to_000011_generated_lip_landmarks.mat")
    parser.add_argument('--fps', type=float, default=25, help="fps of video of landmarks")
    parser.add_argument('--prepend_time_in_ms', '-pptime', type=float, default=200, help="prepend_time_in_ms")
    parser.add_argument('--audio_file_name', '-a', type=str, default=None, help="audio file of lip landmarks; eg. CV_05_C4W1L05_000001_to_000011.wav")
    parser.add_argument('--output_file_name', '-o', type=str, default='lip_landmarks_video.mp4', help="file name of output video")
    parser.add_argument('--ffmpeg_overwrite', '-y', action="store_true")
    parser.add_argument('--verbose', '-v', action="store_true")

    args = parser.parse_args()

    if args.verbose:
        print(args)

    # EXAMPLE: python make_video_of_lip_landmarks.py CV_05_C4W1L05_000001_to_000011_generated_lip_landmarks.mat -a CV_05_C4W1L05_000001_to_000011.wav -o CV_05_C4W1L05_000001_to_000011_lip_landmarks.mp4 -v

    make_video_of_lip_landmarks(lip_landmarks_mat_file=args.lip_landmarks_mat_file, video_fps=args.fps, prepend_time_in_ms=args.prepend_time_in_ms, audio_file_name=args.audio_file_name,
                                output_file_name=args.output_file_name, ffmpeg_overwrite=args.ffmpeg_overwrite, verbose=args.verbose)

