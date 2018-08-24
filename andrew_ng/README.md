# morph_video_with_new_lip_landmarks.py

Provide video, new audio, and new audio's landmarks, and a name for the output video, and this will generate a new video with the lips morphed to match the new audio's landmarks. Example:

```python andrew_ng/morph_video_with_new_lip_landmarks.py <video.mp4> -a <audio.wav> -l <generated_lip_landmarks_from_audio.mat> -o <new_video.mp4>```

OR

```
python /home/voleti.vikram/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py /home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4 -a /home/voleti.vikram/ANDREW_NG/ABHISHEK/audio/CV_01_C4W1L01_000003_to_000045.wav -l /home/voleti.vikram/ANDREW_NG/ABHISHEK/lip_landmarks_mat/CV_01_C4W1L01_000003_to_000045_generated_lip_landmarks.mat -o /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045_hindi_abhishek.mp4 -b -v
```

# NOTE:
Only for dataset created by "make_andrew_ng_dataset.py" (landmarks are detected and saved in appropriate format). Please also refer "exchange_dialogues".

# Options

- *Set options in* ```morph_video_config.py```!
    - ```generator_model_name```: e.g. '/home/voleti.vikram/DeepLearningImplementations/pix2pix/models/20180503_233616_andrew_ng_small/generator_latest.h5'
    - ```dataset_dir```: e.g. '/shared/fusor/home/voleti.vikram/ANDREW_NG'
    - ```person```: e.g. 'andrew_ng'

- ```--save_faces_with_black_mouth_polygons``` or ```-b```

- ```--save_generated_faces``` or ```-f```

- ```--save_both_faces_with_bmp``` or ```-c```

- ```--dont_save_generated_video``` or ```-d```

- ```--verbose``` or ```-v```

# make_andrew_ng_dataset.py

Make a dataset (not only for Andrew Ng!), given a video - detect landmarks, save frames as .jpg files and landmarks in .txt files.

Uses - make_andrew_ng_dataset_every_nth_frame.py => Detect face in every 10th frame, check if the face matched that of actor reference face provided (e.g. Andrew_Ng), if so then detect landmarks in all 10 frames preceeded this frame, continue.

# make_dataset_images_from_landmarks.py

Given video and landmarks for every frame, make a combined image of a frame + the frame with the mouth region replaces by black and lip polygon (for giving input to pix2pix acc. to https://github.com/voletiv/DeepLearningImplementations/tree/master/pix2pix). Check last line in file for example.

