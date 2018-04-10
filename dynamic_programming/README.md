# ALIGNMENT OF SOURCE_VIDEO AND TARGET_AUDIO USING DYNAMIC PROGRAMMING

Given a source video and a target audio, make a new video with its audio as the target audio, and its frames warped from the frames in the source video to match the timing of the target audio.

```python align_new_audio_to_video.py source_video.mp4 target_audio.mp3 new_video.mp4```

OR

```python align_new_audio_to_video.py source_video.mp4 target_audio.wav new_video```


# REQUIREMENTS

- [**ffmpeg**](https://www.ffmpeg.org/) is required to save temporary audio files and new video file.

- subprocess to call ffmpeg from within python

- [speechpy](https://github.com/astorfi/speechpy) to extract MFCC features from .wav audio
    - I installed using ```pip install speechpy```

- [imageio](http://imageio.readthedocs.io/en/stable/) to read mp4 files

- [scipy.io.wavfile](https://docs.scipy.org/doc/scipy-0.14.0/reference/io.html#module-scipy.io.wavfile) to read .wav files

- numpy, os, tqdm


[1] "A Maximum Likelihood Stereo Algorithm" - Ingemar J. Cox, Sunita L. Hingorani, Satish B. Rao [pdf](https://pdfs.semanticscholar.org/b232/e3426e0014389ea05132ea8d08789dcc0566.pdf)

