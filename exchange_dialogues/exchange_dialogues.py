import argparse

from exchange_dialogues_params import *
from exchange_dialogues_functions import *


if __name__ == '__main__':

    config = MovieTranslationConfig()

    parser = argparse.ArgumentParser(description='Exchange dialogues between 2 videos in the MOVIE_TRANSLATION dataset. E.g.: python3 exchange_dialogues.py telugu Mahesh_Babu 47 hindi Aamir_Khan 33')
    parser.add_argument('video1_language', type=str, choices=['english', 'hindi', 'telugu'], help="video1_language: eg. 'telugu'")
    parser.add_argument('video1_actor', type=str, help="video1_actor: eg. 'Mahesh_Babu'")
    parser.add_argument('video1_number', type=int, help="video1_number: eg. '47'")
    parser.add_argument('video2_language', type=str, choices=['english', 'hindi', 'telugu'], help="video2_language: eg. 'telugu'")
    parser.add_argument('video2_actor', type=str, help="video2_actor: eg. 'Mahesh_Babu'")
    parser.add_argument('video2_number', type=int, help="video2_number: eg. '89'")
    parser.add_argument('--output_dir', '-o', type=str, default=os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'exchanged_videos'), help="output_dir to save the videos in: def: '<path/to/MOVIE_TRANSLATION>/exchanged_videos'")

    args = parser.parse_args()

    # # Inputs
    # video1_language, video1_actor, video1_number = "telugu", "Mahesh_Babu", 47
    # video2_language, video2_actor, video2_number = "telugu", "Mahesh_Babu", 89

    # Load generator
    generator_model = load_generator(config.GENERATOR_MODEL_NAME)

    # Exchange
    try:
        new_video1, new_video2 = exchange_dialogues(generator_model,
                                                    video1_language=args.video1_language, video1_actor=args.video1_actor, video1_number=args.video1_number,
                                                    video2_language=args.video2_language, video2_actor=args.video2_actor, video2_number=args.video2_number,
                                                    output_dir=args.output_dir, verbose=True)
    except ValueError as err:
        print(err)
