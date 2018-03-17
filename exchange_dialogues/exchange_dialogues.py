from exchange_dialogues_functions import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('video1_language', type=str, help="video1_language: eg. 'telugu'")
    parser.add_argument('video1_actor', type=str,help="video1_actor: eg. 'Mahesh_Babu'")
    parser.add_argument('video1_number', type=int,help="video1_number: eg. '47'")
    parser.add_argument('video2_language', type=str,help="video2_language: eg. 'telugu'")
    parser.add_argument('video2_actor', type=str,help="video2_actor: eg. 'Mahesh_Babu'")
    parser.add_argument('video2_number', type=int, help="video2_number: eg. '89'")

    args = parser.parse_args()

    # Inputs
    video1_language, video1_actor, video1_number = "telugu", "Mahesh_Babu", 47
    video2_language, video2_actor, video2_number = "telugu", "Mahesh_Babu", 89

    # Load generator
    generator_model = load_generator(GENERATOR_MODEL_NAME)

    # Exchange
    try:
        new_video1, new_video2 = exchange_dialogues(generator_model,
                                                    video1_language=video1_language, video1_actor=video1_actor, video1_number=video1_number,
                                                    video2_language=video2_language, video2_actor=video2_actor, video2_number=video2_number,
                                                    verbose=True)
    except ValueError as err:
        print(err)

    # imageio.mimwrite('output_filename.mp4', new_video1 , fps=24)
    # imageio.mimsave(os.path.join("video1.gif"), new_video1_frames_generated)

