from make_andrew_ng_dataset_functions import*

ANDREW_NG_DIR = '/shared/fusor/home/voleti.vikram/ANDREW_NG/'

# for vid_dir in glob.glob(os.path.join(ANDREW_NG_DIR, 'videos', '*/')):

for vid in glob.glob(os.path.join(ANDREW_NG_DIR, 'videos', 'CV', '*')):
    extract_person_face_frames(vid,
                               out_dir='/shared/fusor/home/voleti.vikram/ANDREW_NG/',
                               person_name='andrew_ng', person_face_image='/shared/fusor/home/voleti.vikram/ANDREW_NG/andrew_ng.png',
                               shape_predictor_path='/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat',
                               face_rec_model_path='/shared/fusor/home/voleti.vikram/dlib_face_recognition_resnet_model_v1.dat',
                               overwrite_frames=True, overwrite_face_shapes=False, save_faces=False)

