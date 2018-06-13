import os

class MovieTranslationConfig():

    if 'voletiv' in os.getcwd():
        # voletiv
        MOVIE_TRANSLATION_DATASET_DIR = '/home/voletiv/Datasets/MOVIE_TRANSLATION/'
        PIX2PIX_CODE_DIR = '/home/voletiv/GitHubRepos/DeepLearningImplementations/pix2pix/'
        YOUTUBE_VIDEOS_DIR = '/home/voletiv/Datasets/MOVIE_TRANSLATION/youtube_videos/'
        SHAPE_PREDICTOR_PATH = '/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat'
        FACE_REC_MODEL_PATH = '/home/voletiv/Downloads/dlib_face_recognition_resnet_model_v1.dat'

    elif 'voleti.vikram' in os.getcwd():
        # voleti.vikram
        MOVIE_TRANSLATION_DATASET_DIR = '/shared/fusor/home/voleti.vikram/MOVIE_TRANSLATION/'
        PIX2PIX_CODE_DIR = '/shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/'
        YOUTUBE_VIDEOS_DIR ='/shared/fusor/home/voleti.vikram/MOVIE_TRANSLATION/youtube_videos/'
        SHAPE_PREDICTOR_PATH = '/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat'
        FACE_REC_MODEL_PATH = '/shared/fusor/home/voleti.vikram/dlib_face_recognition_resnet_model_v1.dat'

    # GENERATOR_MODEL_NAME = os.path.join(PIX2PIX_CODE_DIR, 'models/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5')
    GENERATOR_MODEL_NAME = os.path.join(PIX2PIX_CODE_DIR, 'models/20180606_115113_telugu/generator_latest.h5')

    # To use 'dlib' or 'face_alignment' for landmark detection
    # Check https://github.com/1adrianb/face-alignment for 'face_alignment' installation instructions
    # USING_DLIB_OR_FACE_ALIGNMENT = 'face_alignment'
    USING_DLIB_OR_FACE_ALIGNMENT = 'dlib'
    ENABLE_CUDA = True


# youtube_videos
#     english
#         aBcDeF_gH.mp4

#     hindi
#         aBcDeF_gH.mp4

#     telugu
#         aBcDeF_gH.mp4

# videos
#     english
#         LDC
#             LDC_0000.mp4
#             LDC_0001.mp4
#         GC
#             GC_0000.mp4
#             GC_0001.mp4

#     hindi
#         SK
#         SRK

#     telugu
#         Mahesh_Babu
#         NTR
#         Sharwanand

# metadata
#     english
#         LDC.txt
#         GC.txt

#     hindi
#         SK.txt
#         SRK.txt

#     telugu
#         Mahesh_Babu.txt
#         NTR.txt
#         Sharwanand.txt

# frames
#     english
#         LDC
#             LDC_0000
#                 LDC_0000.gif
#                 LDC_0000_frame_000.png
#                 LDC_0000_frame_001.png
#             LDC_0001
#         GC

#     hindi
#         SK
#         SRK

#     telugu
#         Mahesh_Babu
#         NTR
#         Sharwanand

# landmarks
#     english
#         LDC
#             LDC_0000_landmarks.txt
#             LDC_0000_landmarks.txt
#         GC

#     hindi
#         SK
#         SRK

#     telugu
#         Mahesh_Babu
#         NTR
#         Sharwanand

