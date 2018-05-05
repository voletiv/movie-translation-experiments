# morph_video_with_new_lip_landmarks.py

Provide video, new audio, and new audio's landmarks, and a name for the output video, and this will generate a new video with the lips morphed to match the new audio's landmarks. Example:

```python andrew_ng/morph_video_with_new_lip_landmarks.py <video.mp4> -a <audio.wav> -l <generated_lip_landmarks_from_audio.mat> -o <new_video.mp4>```

OR

```
python /home/voleti.vikram/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py /home/voleti.vikram/ANDREW_NG/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4 -a /home/voleti.vikram/ANDREW_NG/ABHISHEK/audio/CV_01_C4W1L01_000003_to_000045.wav -l /home/voleti.vikram/ANDREW_NG/ABHISHEK/lip_landmarks_mat/CV_01_C4W1L01_000003_to_000045_generated_lip_landmarks.mat -o /home/voleti.vikram/ANDREW_NG/ABHISHEK/videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045_hindi_abhishek.mp4 -b -v
```
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

