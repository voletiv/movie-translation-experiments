from exchange_dialogues_functions import *

# Inputs
video1_language, video1_actor, video1_number = "telugu", "Mahesh_Babu", 47
video2_language, video2_actor, video2_number = "telugu", "Mahesh_Babu", 89

if 'voleti.vikram' in os.getcwd():
    generator_model_name = '/shared/fusor/home/voleti.vikram/DeepLearningImplementations/pix2pix/models/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5'
elif 'voletiv' in os.getcwd():
    generator_model_name = '/home/voletiv/GitHubRepos/pix2pix/models/DeepLearningImplementations/20180314_152941_Mahesh_Babu_black_mouth_polygons/generator_latest.h5'

# Load generator
generator_model = load_generator(generator_model_name)

# Exchange
new_video1, new_video2 = exchange_dialogues(generator_model,
                                            video1_language=video1_language, video1_actor=video1_actor, video1_number=video1_number,
                                            video2_language=video2_language, video2_actor=video2_actor, video2_number=video2_number,
                                            verbose=True)


# imageio.mimwrite('output_filename.mp4', new_video1 , fps=24)
# imageio.mimsave(os.path.join("video1.gif"), new_video1_frames_generated)

