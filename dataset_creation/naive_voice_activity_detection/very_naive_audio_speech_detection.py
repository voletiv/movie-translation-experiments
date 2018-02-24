# http://bastian.rieck.ru/blog/posts/2014/simple_experiments_speech_detection/
import numpy as np
import tqdm

from scipy import stats
from scipy.io import wavfile


def chunks(l, k):
    """
    Yields chunks of size k from a given list.
    """
    for i in range(0, len(l), k):
        yield l[i:i+k]


def shortTermEnergy(frame):
    """
    Calculates the short-term energy of an audio frame. The energy value is
    normalized using the length of the frame to make it independent of said
    quantity.
    """
    return sum( [ abs(x)**2 for x in frame ] ) / len(frame)


def rateSampleByVariation(chunks):
    """
    Rates an audio sample using the coefficient of variation of its short-term
    energy.
    """
    energy = [ shortTermEnergy(chunk) for chunk in chunks ]
    return stats.variation(energy)


def zeroCrossingRate(frame):
    """
    Calculates the zero-crossing rate of an audio frame.
    """
    signs = np.sign(frame)
    signs[signs == 0] = -1
    return len(np.where(np.diff(signs))[0])/len(frame)


def rateSampleByCrossingRate(chunks):
    """
    Rates an audio sample using the standard deviation of its zero-crossing rate.
    """
    zcr = [ zeroCrossingRate(chunk) for chunk in chunks ]
    return np.std(zcr)


def entropyOfEnergy(frame, numSubFrames):
    """
    Calculates the entropy of energy of an audio frame. For this, the frame is
    partitioned into a number of sub-frames.
    """
    lenSubFrame = int(np.floor(len(frame) / numSubFrames))
    shortFrames = list(chunks(frame, lenSubFrame))
    energy      = [ shortTermEnergy(s) for s in shortFrames ]
    totalEnergy = sum(energy)
    energy      = [ e / totalEnergy for e in energy ]
    entropy = 0.0
    for e in energy:
        if e != 0:
            entropy = entropy - e * np.log2(e)
    return entropy


def rateSampleByEntropy(chunks):
    """
    Rates an audio sample using its minimum entropy.
    """
    entropy = [ entropyOfEnergy(chunk, 20) for chunk in chunks ]
    return np.min(entropy)


# Frame size in ms. Will use this quantity to collate the raw samples
# accordingly.
frame_size_in_secs = 0.01
frequency = 44100 # Frequency of the input data
samples_per_frame = int(frequency * frame_size_in_secs)

window_size_in_secs = 0.300
window_number_of_chunks = int(window_size_in_secs*frequency/samples_per_frame)

step_size_in_secs = 0.100
step_number_of_chunks = int(step_size_in_secs*frequency/samples_per_frame)

# DATA

fr, data = wavfile.read("Sharwanand_0004.wav")
if data.shape[1] == 2:
    data = np.mean(data, axis=1)

data = list(data)
chunks_gen = chunks(data, samples_per_frame)
chunked_data = list(chunks_gen)

# variation = rateSampleByVariation(chunked_data)
# zcr = rateSampleByCrossingRate(chunked_data)
# entropy   = rateSampleByEntropy(chunked_data)

signal_variation = []
signal_zcr = []
signal_entropy = []
for i in tqdm.tqdm(range(0, len(chunked_data), step_number_of_chunks)):
    window = chunked_data[i:i+window_number_of_chunks]
    signal_variation.append(rateSampleByVariation(window))
    signal_zcr.append(rateSampleByCrossingRate(window))
    signal_entropy.append(rateSampleByEntropy(window))

np.logical_or(np.logical_or(np.array(signal_variation) >= 1.0, np.array(signal_zcr) >= 0.05), np.array(signal_entropy) < 2.5)


# TO READ AAC FILE
# from pydub import AudioSegment
# sound = AudioSegment.from_file("your/path/to/audio.aac", "aac")

