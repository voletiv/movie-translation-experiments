import argparse
import os

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
    parser.add_argument('--output_dir', '-o', type=str, default=os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, 'exchanged_videos'),
                        help="output_dir to save the videos in: def: '<path/to/MOVIE_TRANSLATION>/exchanged_videos'")
    parser.add_argument('-c', '--enable_cuda_for_face_aligment', action="store_true", help="enable cuda for face aligment (DON'T, if using a generator_model, which you usually would!)")
    parser.add_argument('-g', '--generator_model_name', type=str, default=config.GENERATOR_MODEL_NAME, help="generator model name")
    parser.add_argument('-d', '--using_dlib_or_face_alignment', type=str, default=config.USING_DLIB_OR_FACE_ALIGNMENT, help="using 'dlib' or 'face_alignment'")
    parser.add_argument('--verbose', '-v', action="store_true", help="verbose")

    args = parser.parse_args()

    # # Inputs
    # video1_language, video1_actor, video1_number = "telugu", "Mahesh_Babu", 47
    # video2_language, video2_actor, video2_number = "telugu", "Mahesh_Babu", 89

    # Load generator
    try:
        generator_model = load_generator(args.generator_model_name, verbose=args.verbose)
    except ValueError as err:
        print("\n\n" + str(err) + "\n\n")
        os._exit(0)

    # Exchange
    try:
        new_video1, new_video2 = exchange_dialogues(generator_model,
                                                    video1_language=args.video1_language, video1_actor=args.video1_actor, video1_number=args.video1_number,
                                                    video2_language=args.video2_language, video2_actor=args.video2_actor, video2_number=args.video2_number,
                                                    output_dir=args.output_dir, verbose=args.verbose)

    except KeyboardInterrupt:
        print("\n\nKeyboard Interrupt! (Ctrl+C was pressed.)\n\n")

    except ValueError as err:
        print(err)
