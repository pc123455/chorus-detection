import essentia
import essentia.standard
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import otsu

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

def enhance_sdm(sdm):
    """enhance the self-distance matrix"""
    enhanced_mat = sdm.copy()
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
                enhanced_mat[i, j] += min_value
            else:
                enhanced_mat[i, j] += max_value

    return enhanced_mat

def detect_repetition(sdm, diagonal_num = 30, thres_rate = 0.2):
    """detect repetition, calculate and return the binarized matrix and indeces of candidate diagonals"""

    length = len(sdm)
    dig_mean = np.zeros(length)
    for i in range(0, length):
        dig_mean[i] = np.sum(np.diag(sdm, -i)) / (length - i)

    # using a FIR filter to smooth mean of diagonals
    B = np.ones(50) / 50
    dig_lp = scipy.signal.lfilter(B, 1, dig_mean)
    dig = dig_mean - dig_lp

    # calculate the smoothed differential of diagonals
    B = np.array([1, 0, -1])
    dig_smooth_diiiferentia = scipy.signal.lfilter(B, 1 ,dig)

    # index where the smoothed differential of diagonals from negative to positive
    minima = np.array([])
    minima_indeces = np.array([], dtype = int)
    for i in range(0, len(dig_smooth_diiiferentia) - 1):
        if dig_smooth_diiiferentia[i] < 0 and dig_smooth_diiiferentia[i + 1] > 0:
            minima_indeces = np.append(minima_indeces, i)
            minima = np.append(minima, dig_smooth_diiiferentia[i])

    # delete max value
    if len(minima) > diagonal_num:
        while True:
            add = np.where(minima == max(minima))
            add = add[0 : len(minima) - diagonal_num]
            minima = np.delete(minima, add)
            minima_indeces = np.delete(minima_indeces, add)

            if len(minima) <= diagonal_num:
                break

    # calculate a threshold
    long_vector = np.array([])
    for index in minima_indeces:
        long_vector = np.append(long_vector, np.diag(sdm, -index))

    all_len = len(long_vector)
    long_vector = np.sort(long_vector)
    threshold = long_vector[int(round(thres_rate * all_len))]

    # calculate a binary matrix
    binary_matrix = np.zeros([length, length], dtype = int)

    for index in minima_indeces:
        temp = np.diag(sdm, -index)
        for j in range(0, len(temp)):
            if temp[j] > threshold:
                binary_matrix[index + j, j] = 1



    # enhance the binary matrix
    enhanced_binary_matrix = binary_matrix.copy()
    for index in minima_indeces:
        temp = np.diag(sdm, -index)
        j = 0
        while len(temp) >= 25 or j <= len(temp):
            if temp[j] == 0:
                j += 1
                if j + 25 - 1 > len(temp):
                    break

                continue

            if j + 25 - 1 > len(temp):
                break

            kernel = temp[j : j + 25 - 1]
            if isenhance(kernel):
                for k in range(0, 25):
                    enhanced_binary_matrix[index + j + k] = 1

                j = j + 25 - 1

            j += 1
            if j + 25 - 1 > len(temp):
                break

    return enhanced_binary_matrix, minima_indeces


def isenhance(kernel):
    """determine if a diagonal should be enhanced"""
    length = len(kernel)
    count = 0.0

    # if B(i, j) = 0, this matrix should not be enhanced
    if kernel[0] != 1:
        return False

    for item in kernel:
        if item == 1:
            count += 1

    if count / length >= 0.65 and (kernel[length - 2] == 1 or kernel[length - 1] == 1):
        return True
    else:
        return False


def locate_interesting_segment(binary_matrix, indeces, beats, during_threshold = 4):
    """find the locate interesting segment by binary matrix"""
    point = np.zeros([1, 4], dtype = int)
    segmets = np.empty([0, 4], dtype = int)
    is_segment_bedin = False
    for index in indeces:
        temp = np.diag(binary_matrix, -index)
        for j in range(0, len(temp)):
            if (temp[j] == 0 and is_segment_bedin == False) or (temp[j] == 1 and is_segment_bedin == True):
                continue
            else:
                if temp[j] == 1:
                    point[0, 0] = index + j
                    point[0, 1] = j
                    is_segment_bedin = True
                else:
                    point[0, 2] = index + j
                    point[0, 3] = j
                    is_segment_bedin = False
                    segmets = np.append(segmets, point, axis = 0)

    # using the time during whose default value is 4s to filter segment
    del_indeces = np.array([], dtype = int)
    new_binary_matrix = binary_matrix.copy()
    for i in range(0, len(segmets)):
        time_begin = beats[segmets[i, 0]]
        time_end = beats[segmets[i, 2]]
        if time_end - time_begin < during_threshold:
            del_indeces = np.append(del_indeces, i)

            # set the binary matrix
            for row in range(segmets[i, 0], segmets[i, 2]):
                row_begin = segmets[i, 0]
                col_begin = segmets[i, 1]
                new_binary_matrix[row, row - row_begin + col_begin] = 0

    segmets = np.delete(segmets, del_indeces, axis=0)

    length = len(segmets)
    # the matrix which denote if segment is close with each other
    segmets_close_matrix = np.zeros([length, length], dtype = int)
    for i in range(0, length):
        for j in range(0, length):
            if i == j:
                continue
            x1 = segmets[i, :]
            x2 = segmets[j, :]

            # determine if segment is close with each other
            if x2[0] >= x1[0] - 5 and x2[2] <= x1[2] + 20 and abs(x2[1] - x1[1]) <= 20 and x2[3] <= x1[3] + 5:
                segmets_close_matrix[i, j] = 1

    #delete some segments with less than 3 closed segment
    del_indeces = np.array([], dtype=int)
    close_count = np.sum(segmets_close_matrix, axis = 0)
    for i in range(0, len(segmets)):
        if close_count[i] < 3:
            del_indeces = np.append(del_indeces, i)

            # set the binary matrix
            for row in range(segmets[i, 0], segmets[i, 2]):
                row_begin = segmets[i, 0]
                col_begin = segmets[i, 1]
                new_binary_matrix[row, row - row_begin + col_begin] = 0

    segmets = np.delete(segmets, del_indeces, axis = 0)
    plt.matshow(new_binary_matrix, cmap=plt.cm.gray)
    plt.show()

    return segmets

def otsu_test(matrix):
    img = otsu.quantify(matrix)
    gray = otsu.getGray(img)
    threshold = otsu.getThres(gray)
    otsu.binarize(img, threshold, 1)


# extract audio feature
audio = read_audio('/Users/xueweiyao/Downloads/musics/Madonna - Like a Virgin.wav')
beats, beats_time = extract_beat(audio)
mfcc = extract_mfcc_by_beat(audio, beats)
chroma = extract_chroma_by_beat(audio, beats)

# calculate the self-distance matrix
sdm_chroma = calculate_sdm(chroma)
sdm_mfcc = calculate_sdm(mfcc)



#enhance the self-distance matrix
enhanced_mat = enhance_sdm(sdm_chroma)

# sum the mfcc self-distance matrix and enhanced chroma self-distance matrix
sdm_new = enhanced_mat + sdm_mfcc

# plt.matshow(sdm_new, cmap = plt.cm.gray)
# plt.show()
# otsu_test(sdm_new)

bimar, indeces = detect_repetition(sdm_new)
segment = locate_interesting_segment(bimar, indeces, beats_time)

# plt.matshow(enhanced_mat, cmap=plt.cm.gray)
# plt.matshow(sdm_fcc, cmap=plt.cm.gray)
# plt.matshow(sdm_new, cmap=plt.cm.gray)

#
# plt.plot(dig_mean)
# plt.plot(dig)
# plt.plot(dig_lp)
# plt.plot(dig_smooth_diiiferentia)

