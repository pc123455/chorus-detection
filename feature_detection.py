import essentia
import essentia.standard
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt

def read_audio(filename, sample_rate = 44100):
    """read audio file into a list"""
    loader = essentia.standard.MonoLoader(filename = filename, sampleRate = sample_rate)
    audio = loader()
    return audio

def extract_beat(audio, sample_rate = 44100):
    """extract beats from audio signal list"""
    beat_tracker = BeatTrackerDegara()
    beats_time = beat_tracker(audio)
    beats = np.array(map(lambda time : round(time * sample_rate), beats_time))
    beats = np.append(0, beats)
    beats_time = np.append(0, beats_time)

    return beats, beats_time

def extract_mfcc_by_beat(audio, beats):
    """extrac mfcc by beats from audio"""
    spectrum = Spectrum()
    mfcc = MFCC()
    mfccs = []
    w = Windowing(type = 'square')

    for i in range(0, len(beats) - 1):
        buffer = audio[int(beats[i]) : int(beats[i + 1])]

        if len(buffer) % 2 != 0:
            buffer = np.append(buffer, audio[int(beats[i + 1])])

        frame = w(buffer)
        spec = spectrum(frame)
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)
    
    mfccs = np.matrix(mfccs)
    return mfccs

def extract_chroma_by_beat(audio, beats):
    """extrac chroma by beats from audio"""
    spectrum = Spectrum()
    spec_peak = SpectralPeaks()
    w = Windowing(type = 'square')
    hpcp = HPCP()
    chromas = []

    for i in range(0, len(beats) - 1):
        buffer = audio[int(beats[i]) : int(beats[i + 1])]

        if len(buffer) % 2 != 0:
            buffer = np.append(buffer, audio[int(beats[i + 1])])

        frame = w(buffer)
        spec = spectrum(frame)
        freq, magn = spec_peak(spec)
        chromas.append(hpcp(freq, magn))

    chromas = np.matrix(chromas)

    return chromas

def calculate_sdm(feature_matrix):
    """calculate the self-distance matrix"""
    length = len(feature_matrix)
    self_distance_matrix = np.zeros((length, length))
    for i in range(0, length):
        row1 = feature_matrix[i, :]
        for j in range(0, length):
            row2 = feature_matrix[j, :]
            self_distance_matrix[i, j] = np.sqrt(np.sum(np.square(row1 - row2)))

    return self_distance_matrix

def enhence_sdm(sdm):
    """enhence the self-distance matrix"""
    enhenced_mat = sdm.copy()
    length = len(sdm)
    for i in range(2, length - 2):
        for j in range(2, length - 2):
            up_left = sdm[i - 2, j - 2] + sdm[i - 1, j - 1]
            low_right = sdm[i + 2, j + 2] + sdm[i + 1, j + 1]
            up = sdm[i - 2, j] + sdm[i - 1, j]
            low = sdm[i + 2, j] + sdm[i + 1, j]
            left = sdm[i, j - 2] + sdm[i, j - 1]
            right = sdm[i, j + 2] + sdm[i, j + 1]

            min_value = min([up_left, low_right, up, low, left, right])
            max_value = max([up_left, low_right, up, low, left, right])

            if min_value == up_left or min_value == low_right:
                enhenced_mat[i, j] += min_value
            else:
                enhenced_mat[i, j] += max_value

    return enhenced_mat

def detect_repetition(sdm):
    """detect repetition"""
    length = len(sdm)
    dig = np.zeros([length, 1])
    for i in range(0, length):
        dig[i] = np.sum(np.diag(sdm, -i)) / (length - i)

    return dig


# extract audio feature
audio = read_audio('/Users/xueweiyao/Downloads/musics/Madonna - Like a Virgin.wav')
beats, beats_time = extract_beat(audio)
mfcc = extract_mfcc_by_beat(audio, beats)
chroma = extract_chroma_by_beat(audio, beats)

# calculate the self-distance matrix
sdm_chroma = calculate_sdm(chroma)
sdm_mfcc = calculate_sdm(mfcc)

#enhence the self-distance matrix
enhenced_mat = enhence_sdm(sdm_chroma)

# sum the mfcc self-distance matrix and enhenced chroma self-distance matrix
sdm_new = enhenced_mat + sdm_mfcc

dig = detect_repetition(enhenced_mat)

# plt.matshow(enhenced_mat, cmap=plt.cm.gray)
# plt.matshow(sdm_mfcc, cmap=plt.cm.gray)
# plt.matshow(sdm_new, cmap=plt.cm.gray)
plt.plot(dig)
plt.show()
print "123"