# Automated pipeline of Movie Translation data creation

![alt text](tentative_pipeline.png "IMAGE NOT FOUND")

## INSTALLATION

Requires [webRTC](https://github.com/wiseman/py-webrtcvad)

```
pip2 install webrtcvad
```

## RUN

Example run of `extract_video_clips_via_VAD.py`:

`python2 extract_video_clips_via_VAD.py /path/to/video_file.mp4 [-o /path/to/clips/output/] [--aggressiveness 2] [--verbose]`

or:

`python2 extract_video_clips_via_VAD.py /path/to/audio_file.mp3 [-o /path/to/clips/output/] [-agg 2] [-v]`

Right now, `extract_video_clips_via_VAD.py` conducts the following steps:

- Extract single channel .wav audio from input
    - Save audio as /tmp/my_audio.wav

- Extract speech segments from audio
    - Use [webRTC](https://github.com/wiseman/py-webrtcvad) to extract speech segments from the audio extracted in the previous step
    - This takes the aggressiveness of filtering as an optional argument, given from the command line as `-agg` or `--aggressiveness`

- Write clips
    - Use ffmpeg to clip the input media into segments based on the speech segments extracted in the previous step
    - Save the segments in the directory mentioned after `-o` or `--output-dir` in the command line, or the same folder as the input file if `-o` or `--output-dir` is not mentioned

- Write the details of speech segments extraction into a .txt file
    - Write `output_file_path speech_start_time speech_duration` information, i.e. the name of the clipped file, its starting time in the original file, and its duration, in a .txt file in the output directory (as described in the previous point)

## DRY-RUN

Same steps as above, but don't save the clips. Only save the .txt file with the start times and durations.

