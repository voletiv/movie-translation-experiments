import argparse
import collections
import contextlib
import csv
import os
import subprocess
import sys
import wave
import webrtcvad


def parse_args():
    parser = argparse.ArgumentParser(description='To extract video clips based on Voice Activity Detection')
    parser.add_argument(
        'video_file_path',
        help='path to video file (/path/to/video_file.mp4)',
        default=None,
        type=str
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='video_output_path',
        help='output directory for saving clips. Default: same directory as input video',
        default='same',
        type=str
    )
    parser.add_argument(
        '-a',
        '--audio_file_path',
        dest='audio_file_path',
        help='.wav file path of audio, or to save audio extracted from video. Default: /tmp/my_audio.wav',
        default='/tmp/my_audio.wav',
        type=str
    )
    parser.add_argument(
        '-agg',
        '--aggressiveness',
        dest='aggressiveness',
        help='aggressiveness in filtering out non-speech: integer between 0 and 3, 0 is least, 3 is most. Default: 2',
        default=2,
        type=int
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose',
        help='verbose or not: 1 or 0; Default: 0',
        default=0,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames,
                  verbose=False):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if verbose:
            sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # Note the start time
                speech_start_time = ring_buffer[0][0].timestamp
                if verbose:
                    print("\nMINE", speech_start_time, "\n")
                    sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # Note the duration
                speech_duration = frame.timestamp + frame.duration - speech_start_time
                if verbose:
                    print("\nMINE", frame.timestamp + frame.duration, "\n")
                    sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                # Yield speech, speech_start_time, speech_duration
                yield b''.join([f.bytes for f in voiced_frames]), str(speech_start_time), str(speech_duration)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        # Note the duration
        speech_duration = frame.timestamp + frame.duration - speech_start_time
        if verbose:
            print("\nMINE", frame.timestamp + frame.duration, "\n")
            sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    if verbose:
        sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        # Yield speech, speech_start_time, speech_duration
        yield b''.join([f.bytes for f in voiced_frames]), str(speech_start_time), str(speech_duration)


def extract_audio_from_video(args):
    try:
        # Extract audio using ffmpeg
        # If audio file path is not mentioned, extract audio from video
        if args.audio_file_path is '/tmp/my_audio.wav':
            print("Extracting audio from video using ffmpeg")
            subprocess.call(['ffmpeg', '-loglevel', 'warning', '-i', args.video_file_path, '-y', '-vn', '-ar', str(16000), '-ac', str(1), '-f', 'wav', '/tmp/my_audio.wav'])
        # Else, make audio 16kHz and 1-channel
        else:
            print("Extracting 1-channel audio from audio_file_path using ffmpeg")
            subprocess.call(['ffmpeg', '-loglevel', 'warning', '-i', args.audio_file_path, '-y', '-ar', str(16000), '-ac', str(1), '-f', 'wav', '/tmp/my_audio.wav'])
    except Exception as e:
        print("\nERROR! ffmpeg:", e, "Exiting.\n")
        sys.exit(1)


def extract_speech_segments_from_audio(args):
    # Read audio
    audio, sample_rate = read_wave('/tmp/my_audio.wav')
    # Make VAD object
    vad = webrtcvad.Vad(args.aggressiveness)
    # Collect speech segments
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames, args.verbose)
    # Yield segments = (speech, speech_start_time, speech_duration)
    return segments


def main(args):
    # Extract audio of video, and save in tmp
    # audio_file_name = '/tmp/my_audio.wav'
    extract_audio_from_video(args)
    # Extract speech segments from audio
    segments = extract_speech_segments_from_audio(args)
    # Write video segments
    details = []
    for i, (speech, speech_start_time, speech_duration) in enumerate(segments):
        # path = 'chunk-%002d.wav' % (i,)
        video_file_output_path = os.path.join(args.video_output_path, os.path.splitext(os.path.basename(args.video_file_path))[0] + "_{0:04d}.mp4".format(i))
        details.append([video_file_output_path, speech_start_time, speech_duration])
        print("Using ffmpeg, writing segment from", speech_start_time, "seconds for", speech_duration, "seconds as", video_file_output_path)
        command = ['ffmpeg', '-loglevel', 'warning', '-ss', speech_start_time, '-i', args.video_file_path, '-t', speech_duration, '-y',
                   '-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main', '-acodec', 'aac', '-strict', '-2', video_file_output_path]
        subprocess.call(command)
        # print(' Writing %s' % (path,))
        # write_wave(path, segment, sample_rate)
    # Writing "details" (video_file_output_path, speech_start_time, speech_duration) into csv/txt
    # # Write as csv
    # with open(os.path.join(args.video_output_path, os.path.splitext(os.path.basename(args.video_file_path))[0]) + ".csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(details)
    # Write as txt
    with open(os.path.join(args.video_output_path, os.path.splitext(os.path.basename(args.video_file_path))[0]) + ".txt", "w") as f:
        for row in details:
            line = ""
            for e in row:
                line += e + " "
            line = line[:-1] + "\n"
            f.write(line)


def assert_args(args):
    if args.video_output_path == 'same':
        args.video_output_path = os.path.dirname(args.video_file_path)
    try:
        # Assert video_file_path exists
        assert os.path.exists(args.video_file_path), ("Input file does not exist! Got: " + args.video_file_path)
        # Assert video_file_path is mp4
        assert os.path.splitext(args.video_file_path)[-1] == '.mp4', ("Video file must be .mp4! Got: " + args.video_file_path)
        # Assert audio_file_path is wav
        assert os.path.splitext(args.audio_file_path)[-1] == '.wav', ("audio_file_path must be a .wav file! Got: " + args.audio_file_path)
    except AssertionError as error:
        print('\nERROR:\n', error, '\n')
        print("Exiting.\n")
        sys.exit(1)


if __name__ == '__main__':
    # Read arguments
    args = parse_args()
    # print(args)
    # Assert the arguments
    assert_args(args)
    # print(args)
    # Do your thang
    main(args)
