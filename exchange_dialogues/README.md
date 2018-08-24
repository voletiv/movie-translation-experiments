# exchange\_dialogues

Given video1 and video2 from the MOVIE\_TRANSLATION dataset, make new\_video1 with frames from video1, lips morphed to match the audio of video2, and new\_video2 with frames from video2 but lips morphed to match the audio of video1. Example:

```python3 exchange_dialogues.py telugu Mahesh_Babu 47 hindi Aamir_Khan 33```

(within the directory structure of MOVIE\_TRANSLATION dataset: 'videos' -> language -> actor -> video\_number)

For all options while exchanging dialgues, please see the argparse in exchange\_dialogues.py.

For parameters used, please see exchange\_dialogues\_params.py, for all internal functions please see exchange\_dialogues\_functions.py.

