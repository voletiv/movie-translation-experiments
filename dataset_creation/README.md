## Dataset creation

This code takes metadata files (e.g. Mahesh_Babu.txt) as inputs. Each metadata file contains the following columns separated by space:

- output video file name (e.g. Mahesh_Babu_0000.mp4) | youtube ID (e.g. LS6XiINMc2s) | start time (e.g. 00:08:27.5) | duration (e.g. 00:00:01.5) | dialogue (e.g. EnTi inkA paDukkolEdA)

The example means that the youtube video with ID LS6XiINMc2s (having been saved as LS6XiINMc2s.mp4 in YOUTUBE_VIDEOS_DIR, YOUTUBE_VIDEOS_DIR being mentioned in `movie_translation_data_creation_params.py`) should be cropped from 00:08:27.5 i.e. 8 minutes and 27.5 seconds, for a duration of 00:00:01.5, i.e. 1.5 seconds, and saved as Mahesh_Babu_0000.mp4 in DATASET_DIR (also mentioned in `movie_translation_data_creation_params.py`).

Cropping of videos is done using ffmpeg.

Please specify the relevant parameters in `movie_translation_data_creation_params.py`.

This creates a dataset with the following directory structure inside DATASET_DIR:

videos
    english
        LDC
            LDC_0000.mp4
            LDC_0001.mp4
        GC
            GC_0000.mp4
            GC_0001.mp4

    hindi
        SK
        SRK

    telugu
        Mahesh_Babu
        NTR
        Sharwanand

metadata
    english
        LDC.txt
        GC.txt

    hindi
        SK.txt
        SRK.txt

    telugu
        Mahesh_Babu.txt
        NTR.txt
        Sharwanand.txt

frames
    english
        LDC
            LDC_0000
                LDC_0000.gif
                LDC_0000_frame_000.png
                LDC_0000_frame_001.png
            LDC_0001
        GC

    hindi
        SK
        SRK

    telugu
        Mahesh_Babu
        NTR
        Sharwanand

landmarks
    english
        LDC
            LDC_0000_landmarks.txt
            LDC_0000_landmarks.txt
        GC

    hindi
        SK
        SRK

    telugu
        Mahesh_Babu
        NTR
        Sharwanand


