from exchange_dialogues_functions import *

# Inputs
video1_language, video1_actor, video1_number = "telugu", "Mahesh_Babu", 47
video2_language, video2_actor, video2_number = "telugu", "Mahesh_Babu", 89
generator_model_name = '/shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/models/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5'

# Load generator
generator_model = load_generator(generator_model_name)


