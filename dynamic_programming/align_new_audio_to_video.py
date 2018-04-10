import argparse
import imageio
import numpy as np
import os
import subprocess
import scipy.io.wavefile

from tqdm import tqdm


def cost(a, b):
    # Euclidean distance
    return np.linalg.norm(a - b)


def fix_numbers(y_to_x):
    new_y_to_x = np.array(y_to_x).astype(float)
    start_idx = -1
    for i, num in enumerate(y_to_x[:-1]):
        if y_to_x[i+1] < y_to_x[i]:
            start_idx = i
        elif y_to_x[i+1] > y_to_x[i] and start_idx >= 0:
            end_idx = i + 1
            # print(start_idx, y_to_x[start_idx], end_idx, y_to_x[end_idx])
            new_range = np.linspace(y_to_x[start_idx], y_to_x[end_idx], end_idx - start_idx + 1)[:-1]
            # print(new_range)
            new_y_to_x[start_idx:end_idx] = new_range
            start_idx = -1
    if start_idx >= 0:
        new_y_to_x[start_idx:] = y_to_x[start_idx]
    return new_y_to_x


def dynamic_programming(source, target):
    
    # INITIALIZE VARIABLES
    
    # Init cost
    init_cost = 5

    # Cumulative cost
    C = np.zeros((len(source)+1, len(target)+1))

    for i in range(1, len(source)+1):
        C[i, 0] = i * init_cost

    for j in range(1, len(target)+1):
        C[0, j] = j * init_cost

    # Decider
    M = np.zeros((len(source), len(input_y)))
    
    # Compute cost and note the probable case
    for i in tqdm(range(1, len(source)+1)):
        for j in range(1, len(target)+1):
            min1 = C[i-1, j-1] + cost(source[i-1], target[j-1])
            min2 = C[i-1, j] + init_cost
            min3 = C[i, j-1] + init_cost
            C[i, j] = cmin = min(min1, min2, min3)
            if cmin == min1:
                M[i-1, j-1] = 1
            elif cmin == min2:
                M[i-1, j-1] = 2
            elif cmin == min3:
                M[i-1, j-1] = 3
    
    # MAP 1D SIGNALS USING DYNAMIC PROGRAMMING

    # Track the actual mapping
    mapped_target_frames_of_source_frames = np.zeros((len(source)))
    mapped_source_frames_of_target_frames = np.zeros((len(target)))

    p = len(source)
    q = len(target)

    # Do the mapping
    while p != 0 and q != 0:
        if M[p-1, q-1] == 1:
            # p matches q
            mapped_target_frames_of_source_frames[p-1] = q-1
            mapped_source_frames_of_target_frames[q-1] = p-1
            p -= 1
            q -= 1
        elif M[p-1, q-1] == 2:
            # p is unmatched
            p -= 1
        elif M[p-1, q-1] == 3:
            # q is unmatched
            q -=1

    # FIX THE MAPPING
    
    # The mapping only gives for key frames, the rest frames have '0'
    # fix_numbers interpolates the numbers properly
    mapped_target_frames_of_source_frames_fixed = fix_numbers(mapped_target_frames_of_source_frames)
    mapped_source_frames_of_target_frames_fixed = fix_numbers(mapped_source_frames_of_target_frames)
    
    return mapped_target_frames_of_source_frames_fixed, mapped_source_frames_of_target_frames_fixed
    

def align_new_audio_to_video(source_video, target_dialogue, new_video_name, verbose=False):
    """Dynamic programming reference - "A Maximum Likelihood Stereo Algorithm"
    by Ingemar J. Cox, Sunita L. Hingorani, Satish B. Rao
    (https://pdfs.semanticscholar.org/b232/e3426e0014389ea05132ea8d08789dcc0566.pdf)
    """
    
    # READ SOURCE VIDEO
    if verbose:
        print("Reading source video", source_video)
    video_reader = imageio.get_reader(source_video) 
    video_fps = video_reader.get_meta_data()['fps']
    
    # READ SOURCE AUDIO
    # Convert video's audio into a .wav file
    if verbose:
        print("Writing source video's audio as /tmp/audio.wav")
    subprocess.call(['ffmpeg', '-i', source_video, '-codec:a', 'pcm_s16le', '-ac', '1', '/tmp/audio.wav'])
    # Read the .wav file
    if verbose:
        print("Reading source video's audio - /tmp/audio.wav")
    source_audio_fs, source_audio = scipy.io.wavfile.read('/tmp/audio.wav')
    source_audio = source_audio[:, 0]
    
    # READ TARGET AUDIO
    # Check file type
    file_type = os.path.splitext(target_dialogue)[-1]
    # If file type is not .wav, convert it to .wav and read that
    if file_type != '.wav':
        if verbose:
            print("Target dialogue not a .wav file! Given:", target_dialogue)
            print("Converting target dialogue file into .wav - /tmp/audio.wav")
        subprocess.call(['ffmpeg', '-i', target_dialogue, '-codec:a', 'pcm_s16le', '-ac', '1', '/tmp/audio.wav'])
        target_dialogue = '/tmp/audio.wav'
    # Read the target .wav file
    if verbose:
        print("Reading target audio", target_dialogue)
    target_audio_fs, target_audio = scipy.io.wavfile.read(target_dialogue)
    target_audio = target_audio[:, 0]
    
    # EXTRACT MFCC FEATURES
    frame_length = 0.025
    frame_stride = 0.010
    num_cepstral = 13
    num_filters = 40
    if verbose:
        print("Converting source and target audio into MFCC features with frame_length", frame_length,
              ", frame_stride", frame_stride, ", num_cepstral", num_cepstral, ", num_filters", num_filters)
    # Extract MFCC features of source audio
    source_audio_mfcc = speechpy.feature.mfcc(source_audio, sampling_frequency=source_audio_fs,
                                              frame_length=frame_length, frame_stride=frame_stride,
                                              num_cepstral=num_cepstral, num_filters=num_filters)
    # Extract MFCC features of target audio
    target_audio_mfcc = speechpy.feature.mfcc(target_audio, sampling_frequency=target_audio_fs,
                                              frame_length=frame_length, frame_stride=frame_stride,
                                              num_cepstral=num_cepstral, num_filters=num_filters)
    
    # DO DYNAMIC PROGRAMMING BETWEEN THE SOURCE AND TARGET AUDIO MFCC FRAMES
    if verbose:
        print("Doing dynamic programming between source and target audio")
    mapped_target_audio_frames_of_source_audio_frames, \
        mapped_source_audio_frames_of_target_audio_frames = dynamic_programming(source_audio_mfcc, target_audio_mfcc)
    
    # CONVERT AUDIO MAPPING TO VIDEO MAPPING, i.e. mapped_source_video_frames_of_target_video_frames
    if verbose:
        print("Converting mapped_source_audio_frames_of_target_audio_frames into mapped_source_video_frames_of_target_video_frames")
    # Get source videos frames of the target audio frames
    mapped_source_video_frames_of_target_audio_frames = mapped_source_audio_frames_of_target_audio_frames * frame_stride * video_fps
    # Calculate the number of target video frames (from the number of audio frames and fps)
    num_of_target_video_frames = round( len(target_audio_mfcc) * frame_stride * video_fps )
    # Make a linear mapping from the target audio frames to target video frames
    target_audio_frames_idx_of_target_video_frames = np.round(np.linspace(0,
                                                                          len(target_audio_mfcc)-1,
                                                                          num_of_video_frames)).astype(int)
    # Select the source video frames corresponding to each target video frame
    mapped_source_video_frames_of_target_video_frames = np.floor(mapped_source_video_frames_of_target_audio_frames[target_audio_frames_idx_of_target_video_frames]).astype(int)
    
    # MAKE NEW VIDEO
    if verbose:
        print("Making new video", new_video_name)
    # Read video
    source_frames = []
    for frame in video_reader:
        source_frames.append(frame)
    # Note new frames
    new_frames = []
    for source_frame_number in mapped_source_video_frames_of_target_video_frames:
        new_frames.append(source_frames[int(source_frame_number)])
    # Save new video
    if os.path.splitext(new_video_name)[-1] != '.mp4':
        new_video_name += '.mp4'
        if verbose:
            print("new_video_name not mp4! Modified to", new_video_name)
    
    if verbose:
        print("Writing mp4 of new video frames /tmp/video.mp4")
    imageio.mimwrite('/tmp/video.mp4', new_frames, fps=video_fps)
    
    if verbose:
        print("Writing new video with source_video frames and target dialogue", new_video_name)
    command = ['ffmpeg', '-loglevel', 'error',
               '-i', '/tmp/video.mp4', '-i', target_dialogue, '-y',
               '-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main',
               '-acodec', 'aac', '-strict', '-2',
               new_video_name]
    subprocess.call(command)
    
    if verbose:
        print("Done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Exchange dialogues between 2 videos in the MOVIE_TRANSLATION dataset. E.g.: python align_new_audio_to_video.py source_video.mp4 target_audio.mp3 new_video.mp4')
    parser.add_argument('source_video', type=str, help="name of source video, a .mp4 file: eg. 'source_video.mp4'")
    parser.add_argument('target_audio', type=str, help="name of target audio, a .wav or .mp3 file: eg. 'source_audio.mp3', or 'source_audio.wav'")
    parser.add_argument('new_video_name', type=int, help="name of new video, a .mp4 file: eg. 'new_video.mp4'")
    parser.add_argument('--verbose', '-v', action="store_true", help="verbose")

    args = parser.parse_args()

    align_new_audio_to_video(source_video, target_audio, new_video_name, verbose)

