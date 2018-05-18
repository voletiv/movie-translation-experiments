from __future__ import print_function

from movie_translation_data_creation_functions import *

using_dlib_or_face_alignment = 'face_alignment'

dlib_detector = None
dlib_predictor = None
face_alignment_3D_object = utils.load_face_alignment_object(d='3D', enable_cuda=True)
face_alignment_2D_object = utils.load_face_alignment_object(d='2D', enable_cuda=True)

for video_file in sorted(glob.glob('/ssd_scratch/cvit/isha/VIKRAM/videos/english/andrew_ng/*.mp4')):
    extract_face_frames_and_landmarks_from_video(video_file, using_dlib_or_face_alignment,
                                                 dlib_detector=dlib_detector, dlib_predictor=dlib_predictor,
                                                 face_alignment_3D_object=face_alignment_3D_object, face_alignment_2D_object=face_alignment_2D_object,
                                                 crop_expanded_face_square=True, resize_to_shape=(256, 256),
                                                 save_with_blackened_mouths_and_polygons=True,
                                                 save_landmarks_as_txt=True, save_landmarks_as_csv=False,
                                                 skip_frames=0, check_for_face_every_nth_frame=10,
                                                 output_dir='/ssd_scratch/cvit/isha/VIKRAM/ANDREW_NG_LS3D',
                                                 verbose=True)

