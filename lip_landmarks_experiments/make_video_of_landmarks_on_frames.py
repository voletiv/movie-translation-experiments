import argparse
import imageio
import numpy as np
import os
import tqdm

import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import utils


def make_video_of_landmarks_on_frames(landmarks_file, video_file_name, audio_file_name=None, output_file_name='lip_landmarks_video.mp4', ffmpeg_overwrite=False, verbose=False):

    landmarks = utils.read_landmarks_list_from_txt(landmarks_file)
    landmarks = np.array([lms[1:] for lms in landmarks])

    reader = imageio.get_reader(video_file_name)
    video_fps = reader.get_meta_data()['fps']

    new_frames = []
    tmp_image_name = '/tmp/%s_lm.png' % os.path.splitext(os.path.basename(video_file_name))[0]

    for frame, landmarks_in_frame in tqdm.tqdm(zip(reader, landmarks), total=min(len(reader), len(landmarks))):
        utils.plot_2D_landmarks(frame, landmarks_in_frame, save_or_show='save', fig_name=tmp_image_name)
        new_frames.append(imageio.imread(tmp_image_name))

    if audio_file_name is None:
        audio_file_name = video_file_name

    utils.save_new_video_frames_with_target_audio_as_mp4(new_frames, video_fps, audio_file_name,
                                                         output_file_name=output_file_name,
                                                         overwrite=ffmpeg_overwrite, verbose=verbose)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make video of lip landmarks')
    parser.add_argument('landmarks_file', type=str, help="landmarks_txt_file (.txt); eg. CV_05_C4W1L05_000001_to_000011_landmarks.txt")
    parser.add_argument('video_file_name', type=str, default=None, help="video on which to overlay landmarks")
    parser.add_argument('--audio_file_name', '-a', type=str, default=None, help="audio file of lip landmarks; eg. CV_05_C4W1L05_000001_to_000011.wav")
    parser.add_argument('--output_file_name', '-o', type=str, default='lip_landmarks_video.mp4', help="file name of output video")
    parser.add_argument('--ffmpeg_overwrite', '-y', action="store_true")
    parser.add_argument('--verbose', '-v', action="store_true")

    args = parser.parse_args()

    if args.verbose:
        print(args)

    # EXAMPLE: python make_video_of_landmarks_on_frames.py CV_05_C4W1L05_000001_to_000011_landmarks.txt CV_05_C4W1L05_000001_to_000011.mp4 -a CV_05_C4W1L05_000001_to_000011.wav -o CV_05_C4W1L05_000001_to_000011_lip_landmarks.mp4 -v

    # EXAMPLE: python make_video_of_landmarks_on_frames.py /home/voleti.vikram/ANDREW_NG_CLIPS/landmarks_in_frames_person/CV_05_C4W1L05_000837_to_000901_CV_05_C4W1L05_000837_to_000901_landmarks_in_frames_andrew_ng.txt /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_05_C4W1L05_000837_to_000901/CV_05_C4W1L05_000837_to_000901.mp4 -a /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_05_C4W1L05_000837_to_000901/CV_05_C4W1L05_000837_to_000901_hindi_abhishek.wav -o CV_05_C4W1L05_000837_to_000901/CV_05_C4W1L05_000837_to_000901_landmarks.mp4 -v

    make_video_of_landmarks_on_frames(landmarks_file=args.landmarks_file, video_file_name=args.video_file_name, audio_file_name=args.audio_file_name,
                                      output_file_name=args.output_file_name, ffmpeg_overwrite=args.ffmpeg_overwrite, verbose=args.verbose)

