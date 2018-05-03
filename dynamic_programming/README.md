# ALIGNMENT OF SOURCE_VIDEO AND TARGET_AUDIO USING DYNAMIC PROGRAMMING

Given a source video and a target audio, make a new video with its audio as the target audio, and its frames warped from the frames in the source video to match the timing of the target audio.

```python align_new_audio_to_video.py source_video.mp4 target_audio.mp3 new_video.mp4```

OR

```python align_new_audio_to_video.py source_video.mp4 target_audio.wav new_video.mp4```


# REQUIREMENTS

- [**ffmpeg**](https://www.ffmpeg.org/) is required to save temporary audio files and new video file.

- subprocess to call ffmpeg from within python

- [speechpy](https://github.com/astorfi/speechpy) to extract MFCC features from .wav audio
    - I installed using ```pip install speechpy```

- [imageio](http://imageio.readthedocs.io/en/stable/) to read mp4 files

- [scipy.io.wavfile](https://docs.scipy.org/doc/scipy-0.14.0/reference/io.html#module-scipy.io.wavfile) to read .wav files

- numpy, os, tqdm

[1] "A Maximum Likelihood Stereo Algorithm" - Ingemar J. Cox, Sunita L. Hingorani, Satish B. Rao [pdf](https://pdfs.semanticscholar.org/b232/e3426e0014389ea05132ea8d08789dcc0566.pdf)

# METHOD

Given a source_video and a target_audio, make a new video with its audio as target_audio, and the frames warped from source_video such that they match target_audio's timing. The mapping of audio and frames is done via dynamic programming.

- Extract part of Andrew Ng's tutorial video using ffmpeg, and save in appropriate file

```ffmpeg -ss <start_time> -i <video_to_clip.mp4> -t <duration> -vcodec libx264 -preset ultrafast -profile:v main -acodec aac -strict -2 <output_video_name.mp4>```

Example:
```ffmpeg -ss 00:07:19 -i /home/voleti.vikram/ANDREW_NG/videos/CV/03.C4W1L03\ More\ Edge\ Detection.mp4 -t 00:00:38 -vcodec libx264 -preset ultrafast -profile:v main -acodec aac -strict -2 /home/voleti.vikram/ANDREW_NG/videos/CV_03_C4W1L03_000719_to_000757/CV_03_C4W1L03_000719_to_000757.mp4```

- Record the same dialogues by someone else - CV_04_C4W1L04_000922_to_000949_ma.mp3

- Use dynamic programming to change video timing to match audio

```python movie-translation-experiments/dynamic_programming/align_new_audio_to_video.py CV_04_C4W1L04_000922_to_000949.mp4 CV_04_C4W1L04_000922_to_000949_ma.mp3 CV_04_C4W1L04_000922_to_000949_ma.mp4```

# MISC

To overlay audio over video (naive dubbing):

```ffmpeg -i /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_02_C4W1L02_000006_to_000013/CV_02_C4W1L02_000006_to_000013.mp4 -i CV_02_C4W1L02_000006_to_000013_pauline_english.m4a -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 /shared/fusor/home/voleti.vikram/ANDREW_NG/videos/CV_02_C4W1L02_000006_to_000013/CV_02_C4W1L02_000006_to_000013_pauline_english_dub.mp4```

