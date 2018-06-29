import argparse
import os

import sys
sys.path.append(os.parth.realpath(os.path.join(os.path.dirname(__file__), '../')
import utils

from movie_translation_data_creation_functions import extract_face_frames_and_landmarks_from_video


def create_dataset_for_video(video_file, using_dlib_or_face_alignment,
                             dlib_detector=None, dlib_predictor=None,
                             face_alignment_3D_object=None, face_alignment_2D_object=None, enable_cuda=True,
                             crop_expanded_face_square=True, resize_to_shape=(256, 256),
                             save_with_blackened_mouths_and_polygons=True,
                             save_landmarks_as_txt=True, save_landmarks_as_csv=False,
                             skip_frames=0, check_for_face_every_nth_frame=10,
                             output_dir=os.path.realpath('VIDEO_DATASET'),
                             verbose=False):

    # Load landmarks detector
    if config.USING_DLIB_OR_FACE_ALIGNMENT == 'dlib':
        dlib_detector, dlib_predictor = utils.load_dlib_detector_and_predictor()
    elif config.USING_DLIB_OR_FACE_ALIGNMENT == 'face_alignment':
        face_alignment_3D_object = utils.load_face_alignment_object(d='3D', enable_cuda=enable_cuda)
        face_alignment_2D_object = utils.load_face_alignment_object(d='2D', enable_cuda=enable_cuda)

    # Make dataset
    extract_face_frames_and_landmarks_from_video(video_file=video_file, using_dlib_or_face_alignment=using_dlib_or_face_alignment,
                                                 dlib_detector=dlib_detector, dlib_predictor=dlib_predictor,
                                                 face_alignment_3D_object=face_alignment_3D_object, face_alignment_2D_object=face_alignment_2D_object,
                                                 crop_expanded_face_square=crop_expanded_face_square, resize_to_shape=resize_to_shape,
                                                 save_with_blackened_mouths_and_polygons=save_with_blackened_mouths_and_polygons,
                                                 save_landmarks_as_txt=save_landmarks_as_txt, save_landmarks_as_csv=save_landmarks_as_csv,
                                                 skip_frames=skip_frames, check_for_face_every_nth_frame=check_for_face_every_nth_frame,
                                                 output_dir=output_dir, verbose=verbose)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Morph lips in input video with target lip landmarks')
    parser.add_argument('video_file', type=str, help="video (.mp4); eg. /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4")
    parser.add_argument('--using_dlib_or_face_alignment', type=str, choices=["dlib", "face_alignment"], default="face_alignment", help="Choose dlib or face_alignment to detect landmarks, IF detect_landmarks_in_video is True")
    parser.add_argument('--disable_cuda', '-dcuda', action="store_true")
    parser.add_argument('--output_dir', '-o', type=str, default=os.path.realpath('.'), help="Name of the directory to make in which to put all faces_combined and landmarks dirs")
    parser.add_argument('--verbose', '-v', action="store_true")

    create_dataset_for_video(video_file=args.video_file,
                             using_dlib_or_face_alignment=args.using_dlib_or_face_alignment, enable_cuda=(not args.disable_cuda),
                             output_dir=args.output_dir,
                             verbose=args.verbose)

