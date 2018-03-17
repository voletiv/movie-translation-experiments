class MovieTranslationConfig():

    if 'voletiv' in os.getcwd():
        # voletiv
        MOVIE_TRANSLATION_DATASET_DIR = '/home/voletiv/Datasets/MOVIE_TRANSLATION/'
        GENERATOR_MODEL_NAME = '/home/voletiv/GitHubRepos/pix2pix/models/DeepLearningImplementations/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5'

    elif 'voleti.vikram' in os.getcwd():
        # voleti.vikram
        MOVIE_TRANSLATION_DATASET_DIR = '/shared/fusor/home/voleti.vikram/MOVIE_TRANSLATION/'
        GENERATOR_MODEL_NAME = '/shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/models/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5'
