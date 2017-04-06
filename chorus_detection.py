#-*- coding: UTF-8 -*-
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

    for i in range(len(beats) - 1):
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

    for i in range(len(beats) - 1):
        buffer = audio[int(beats[i]) : int(beats[i + 1])]

        if len(buffer) % 2 != 0:
            buffer = np.append(buffer, audio[int(beats[i + 1])])

        frame = w(buffer)
        spec = spectrum(frame)
        freq, magn = spec_peak(spec)
        chromas.append(hpcp(freq, magn))

    chromas = np.matrix(chromas)

    return chromas

def calculate_sdm(feature_matrix, is_normalization = False):
    """calculate the self-distance matrix"""
    length = len(feature_matrix)
    self_distance_matrix = np.zeros((length, length))
    for i in range(length):
        row1 = feature_matrix[i, :]
        for j in range(length):
            row2 = feature_matrix[j, :]
            self_distance_matrix[i, j] = np.sqrt(np.sum(np.square(row1 - row2)))

    if is_normalization:
        minima = np.min(self_distance_matrix)
        maxima = np.max(self_distance_matrix)
        self_distance_matrix = (self_distance_matrix - minima) / (maxima - minima)
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
    # dig_mean = np.zeros(length)
    # for i in range(length):
    #     dig_mean[i] = np.sum(np.diag(sdm, -i)) / (length - i)
    dig_mean = calculate_sdm_min_diagonal(sdm, window_size = 48, is_partial = False)

    # using a FIR filter to smooth mean of diagonals
    B = np.ones(50) / 50
    dig_lp = scipy.signal.lfilter(B, 1, dig_mean)
    dig = dig_mean - dig_lp

    # calculate the smoothed differential of diagonals
    B = np.array([1, 0, -1])
    dig_smooth_diiiferentia = scipy.signal.lfilter(B, 1 ,dig)

    plt.plot(dig_mean)
    plt.plot(dig)
    plt.plot(dig_lp)
    plt.plot(dig_smooth_diiiferentia)

    # index where the smoothed differential of diagonals from negative to positive
    # the minima value is the minimum value of diagonals
    minima = np.array([])
    minima_indeces = np.array([], dtype = int)
    for i in range(len(dig_smooth_diiiferentia) - 1):
        if dig_smooth_diiiferentia[i] < 0 and dig_smooth_diiiferentia[i + 1] > 0:
            minima_indeces = np.append(minima_indeces, i)
            minima = np.append(minima, dig[i])

    # delete by otsu algorithm
    threshold_otsu = get_otsu_threshold(np.matrix(minima))
    del_indeces = np.array([])
    # for i in range(len(minima)):
    #     if minima[i] > threshold_otsu:
    #         del_indeces = np.append(del_indeces, i)

    while True:
        threshold_otsu += 1
        del_indeces = np.array([])
        for i in range(len(minima)):
            if minima[i] > threshold_otsu:
                del_indeces = np.append(del_indeces, i)

        if len(minima_indeces) - len(del_indeces) > 50:
            break




    minima = np.delete(minima, del_indeces)
    minima_indeces = np.delete(minima_indeces, del_indeces)

    # calculate a threshold
    long_vector = np.array([])
    for index in minima_indeces:
        long_vector = np.append(long_vector, np.diag(sdm, -index))

    all_len = len(long_vector)
    long_vector = np.sort(long_vector)

    while(True):

        threshold = long_vector[int(round(thres_rate * all_len))]
        minima_count = 0

        # calculate a binary matrix
        binary_matrix = np.zeros([length, length], dtype = int)


        for index in minima_indeces:
            temp = np.diag(sdm, -index)
            for j in range(len(temp)):
                if temp[j] > threshold:
                    binary_matrix[index + j, j] = 1
                    minima_count += 1

        # if the number of segments is smaller than 10
        if minima_count < 20 and thres_rate < 1:
            thres_rate += 0.05
        else:
            break


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
                for k in range(25):
                    enhanced_binary_matrix[index + j + k] = 1

                j = j + 25 - 1

            j += 1
            if j + 25 - 1 > len(temp):
                break

    return enhanced_binary_matrix, minima_indeces

def calculate_sdm_min_diagonal(sdm, window_size = 48, is_partial = True):
    """calculate the min diagonal segment of self-distance matrix"""
    length = len(sdm)
    dig_mean = np.zeros(length)
    for i in range(length):
        diag = np.diag(sdm, -i)
        if is_partial and len(diag) > window_size:
            window = np.ones(window_size) / window_size
            dig_mean[i] = np.min(np.convolve(diag, window, mode = 'valid'))
        else:
            dig_mean[i] = np.sum(np.diag(sdm, -i)) / (length - i)

    return dig_mean

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

def locate_interesting_segment(binary_matrix, indeces, beats, during_threshold = 5):
    """find the locate interesting segment by binary matrix"""
    point = np.zeros([1, 4], dtype = int)
    segments = np.empty([0, 4], dtype = int)
    is_segment_bedin = False
    for index in indeces:
        temp = np.diag(binary_matrix, -index)
        for j in range(len(temp)):
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
                    segments = np.append(segments, point, axis = 0)

    # using the time during whose default value is 4s to filter segment
    del_indeces = np.array([], dtype = int)
    new_binary_matrix = binary_matrix.copy()
    for i in range(len(segments)):

        time_begin = beats[segments[i, 0]]
        time_end = beats[segments[i, 2]]
        if time_end - time_begin < during_threshold:
            del_indeces = np.append(del_indeces, i)

            # set the binary matrix
            for row in range(segments[i, 0], segments[i, 2]):
                row_begin = segments[i, 0]
                col_begin = segments[i, 1]
                new_binary_matrix[row, row - row_begin + col_begin] = 0

    segments = np.delete(segments, del_indeces, axis=0)

    length = len(segments)
    # the matrix which denote if segment is close with each other
    segments_close_matrix = np.zeros([length, length], dtype = int)
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            x1 = segments[i, :]
            x2 = segments[j, :]

            # determine if segment is close with each other
            if x2[0] >= x1[0] - 5 and x2[2] <= x1[2] + 20 and abs(x2[1] - x1[1]) <= 20 and x2[3] <= x1[3] + 5:
                segments_close_matrix[i, j] = 1

    # delete some segments with less than 3 closed segment
    del_indeces = np.array([], dtype=int)
    close_count = np.sum(segments_close_matrix, axis = 0)
    for i in range(len(segments)):
        if close_count[i] < 3:
            del_indeces = np.append(del_indeces, i)

            # set the binary matrix
            for row in range(segments[i, 0], segments[i, 2]):
                row_begin = segments[i, 0]
                col_begin = segments[i, 1]
                new_binary_matrix[row, row - row_begin + col_begin] = 0

    segments = np.delete(segments, del_indeces, axis = 0)
    # plt.matshow(new_binary_matrix, cmap=plt.cm.gray)
    # plt.show()

    return segments, new_binary_matrix

def otsu_test(matrix):
    img = otsu.quantify(matrix)
    gray = otsu.getGray(img)
    threshold = otsu.getThres(gray)
    otsu.binarize(img, threshold, 1)

def get_otsu_threshold(matrix, depth = 8):
    maximum = np.max(matrix)
    minimum = np.min(matrix)
    img = matrix.copy()
    img = (img - minimum) / (maximum - minimum) * (pow(2, depth) - 1)

    gray = otsu.getGray(img)
    threshold = otsu.getThres(gray)

    threshold = float(threshold) / (pow(2, 8) - 1) * (maximum - minimum) + minimum
    return threshold

def calculate_segments_scores(sdm, binary_matrix, segments, audio, beats):
    # M is the length of the song beats
    M = float(len(binary_matrix))

    # s_1 and s_2
    length = len(segments)
    scores = np.zeros([length, 2])
    for i in range(length):
        delta_x = float(segments[i, 2] - segments[i, 0])

        # calculate s_1-score
        # s_1-score measures the difference of the middle column of segment x_p[i, j, i', j'] to one quarter of song length
        scores[i, 0] = 1 - np.abs((segments[i, 1] + delta_x / 2) - round(M / 4)) / round(M / 4)

        # calculate s_2-score
        # s_2-score measures the difference of the middle row of segment x_p to three quarters of the song length
        scores[i, 1] = 1 - np.abs((segments[i, 0] + delta_x / 2) - round(3 * M / 4)) / round(M / 4)

    # s_3
    # s_3_scores = calculate_s3(segments)
    s_3_scores = np.zeros([len(segments), 1])
    scores = np.concatenate((scores, s_3_scores), axis = 1)

    # s_4 and s_5
    average_energy = np.mean(np.square(audio))
    average_distance = np.mean(sdm)
    s_4_and_5_score = np.zeros([length, 2])
    for i in range(length):
        average = get_average_energy(audio, beats, segments[i, 0], segments[i, 2])
        #average -= average_energy
        s_4_and_5_score[i, 0] = average - average_energy

        median_distance = get_median_distance(sdm, segments[i, :])
        s_4_and_5_score[i, 1] = 1 - median_distance / average_distance

    scores = np.concatenate((scores, s_4_and_5_score), axis = 1)

    s_6_score = calculate_s6(segments)

    scores = np.concatenate((scores, s_6_score), axis = 1)

    return scores

def select_segment_most_likely_chorus(scores, beats_time, segments):
    """calculate the final score of segments"""
    length = len(scores)
    final_score = np.zeros(length)
    for i in range(length):
        final_score[i] = 0.5 * (scores[i, 0] + scores[i, 1] + scores[i, 3] + scores[i, 5]) + scores[i, 2] + scores[i, 4]

    best = segments[np.argmax(final_score)]
    return final_score, best

def calculate_s3(segments):
    """calculate the s_3-score of segments
    """
    groups = np.empty([0, 4])
    group = np.zeros([1, 4])
    length = len(segments)
    for i in range(length):
        x_u = segments[i, :]
        for j in range(length):
            if j == i:
                continue
            x_b = segments[j, :]

            # if either x_b is not below x_u, or there is not some overlap between x_b and x_u, x_b will be skipped.
            if x_b[0] <= x_u[2] or x_b[1] > x_u[3]:
                continue

            for k in range(length):
                if k == i or k == j:
                    continue
                x_r = segments[k, :]

                # if either x_r is not right x_b, or there is not some overlap between x_r and x_b, x_r will be skipped.
                if x_r[1] <= x_b[3] or x_r[0] > x_b[2]:
                    continue

                group[0, 0] = i
                group[0, 1] = j
                group[0, 2] = k

                groups = np.append(groups, group, axis = 0)

    # calculate the score of all groups
    # structure of groups row -> | x_u | x_b | x_r | sigma_hat_z |
    group_len = len(groups)
    for i in range(group_len):
        x_u = segments[int(groups[i, 0]), :]
        x_b = segments[int(groups[i, 1]), :]
        x_r = segments[int(groups[i, 2]), :]

        delta_x_u = float(x_u[2] - x_u[0])
        delta_x_b = float(x_b[2] - x_b[0])
        delta_x_r = float(x_r[2] - x_r[0])

        # sigma_1-score measures how close is the end point of the above segment x_u and below segment x_b
        sigma_1 = 1 - 2 * np.abs(x_u[3] - x_b[3]) / (delta_x_b + delta_x_u)

        # sigma_2-score depends on the vertical alignment of upper and below segments
        sigma_2 = 1
        if x_b[1] < x_u[1]:
            sigma_2 = 1 - (x_u[1] - x_b[1]) / delta_x_b
        elif x_b[1] > x_u[3]:
            sigma_2 = 1 - (x_b[1] - x_u[3]) / delta_x_b

        # sigma_3-score measures whether the segments x_b and x_r are of equal length
        sigma_3 = 1 - np.abs(delta_x_r - delta_x_b) / delta_x_b

        # sigma_4-score depends on the difference in the position of left and right segments
        sigma_4 = 1- 2 * np.min([np.abs(x_b[0] - x_r[0]), np.abs(x_b[2] - x_r[2])]) / (delta_x_r + delta_x_b)

        # sigma_hat_z = np.mean([sigma_1, sigma_2, sigma_3, sigma_4])
        # groups[i, 3] = sigma_hat_z
        groups[i, 3] = np.mean([sigma_1, sigma_2, sigma_3, sigma_4])

    #calculate the final s3_score of each segment
    s_3_scores = np.zeros([length, 1])
    for i in range(group_len):
        b_index = int(groups[i, 1])
        s_3 = groups[i, 3]
        if s_3_scores[b_index, 0] < s_3:
            s_3_scores[b_index, 0] = s_3

    return s_3_scores

def calculate_s6(segments):
    length = len(segments)
    repetition_counts = np.zeros([length, 1])
    for i in range(length):
        x_p = segments[i, :]
        for j in range(length):
            if i == j:
                continue

            x_q = segments[j, :]
            delta_x_q = x_q[2] - x_q[0]
            if np.abs(x_p[1] - x_q[1]) <= 0.2 * delta_x_q and np.abs(x_p[3] - x_q[3]) <= 0.2 * delta_x_q:
                repetition_counts[i, 0] += 1

    s_6_scores = repetition_counts / np.max(repetition_counts)

    return s_6_scores

def get_average_energy(audio, beats, begin, end):
    """calculate the average energy of audio segment"""
    buffer = np.square(audio[int(beats[int(begin)]):int(beats[int(end)])])
    average = np.mean(buffer)
    return average

def get_median_distance(sdm, segment):
    """calculate the median distance of self-distance matrix"""
    r_1 = segment[0]
    c_1 = segment[1]
    r_2 = segment[2]
    c_2 = segment[3]

    return np.median(np.diag(sdm[r_1 : r_2, c_1 : c_2]))

def get_segment_time(segment, beats_time):
    time = np.zeros(4)
    for i in range(4):
        time[i] = beats_time[segment[i]]

    return time

def find_location_of_chorus(segments, sdm, time_len = (48, 64, 96)):
    segment = segments.copy()

    M = len(sdm)
    length = segment[2] - segment[0]
    rho_32 = np.zeros([length, 2])
    rho_64 = np.zeros([length, 2])
    for i in range(length):
        x = segment + i
        rho_32[i] = filter_2d(x, sdm, time_len[0])
        rho_64[i] = filter_2d(x, sdm, time_len[2])

    rho_min_32 = np.min(rho_32, axis = 0)
    rho_min_64 = np.min(rho_64, axis = 0)

    chorus_begin = segment[1]
    chorus_end = segment[3]

    # if min_roh_alpha_64 < min_roh_alpha_32, indicates a good match with the 64 beat long chorus with two 32 beat long subsections
    if rho_min_64[0] < rho_min_32[0]:
        chorus_begin = segment[1] + np.argmin(rho_64[:, 0], axis = 0)
        chorus_end = np.min([chorus_begin + time_len[2], M])

    elif length < time_len[0]:
        chorus_begin = segment[1] + np.argmin(rho_32[:, 0], axis = 0)
        chorus_end = np.min([chorus_begin + time_len[0], M])

    elif np.abs(length - time_len[1]) < np.abs(length - time_len[0]) and np.abs(length - time_len[1]) < np.abs(length - time_len[2]) \
            and rho_min_32[0] < rho_min_64[0] and rho_min_32[1] < rho_min_32[1] \
            and np.argmin(rho_32[:, 0]) == np.argmin(rho_32[:, 1]):
        chorus_begin = segment[1] + np.argmin(rho_32[:, 0])
        chorus_end = np.min([chorus_begin + time_len[0], M])

    else:
        segment[0] = np.max([1, segment[0] - 5])
        segment[1] = np.min([M, segment[1] + 5])
        segment[2] = np.max([1, segment[2] - 5])
        segment[3] = np.min([M, segment[3] + 5])

        rate = filter_1d(x, sdm)
        if np.min(rate < 0.7) and np.abs(length - time_len[0]) < np.abs(length - time_len[2]):
            chorus_begin = segment[1] + np.argmin(rate)
            chorus_end = np.min([chorus_begin + time_len[0], M])

        elif length > time_len[1]:
            chorus_begin = segment[1] + np.argmin(rate)
            chorus_end = np.min([chorus_begin + time_len[0], M])

    return chorus_begin, chorus_end

def filter_2d(x, sdm, size):
    """calculate the rho rate for a point in self-distance matrix"""
    col_begin = np.max([1, x[1] - int(size / 2)]) - 1
    col_end = np.min([x[1] + int(size / 2), len(sdm)]) - 1
    row_begin = np.max([1, x[0] - int(size / 2)]) - 1
    row_end = np.min([x[0] + int(size / 2), len(sdm)]) - 1

    beats_count = np.min([row_end - row_begin, col_end - col_begin])

    area = sdm[row_begin : row_begin + beats_count, col_begin : col_begin + beats_count]

    # the main diagonal
    main_diag = np.diag(area, 0)

    # black diagonals
    diags = np.concatenate((main_diag, np.diag(area, -int(size / 2)), np.diag(area, int(size / 2))))

    _alpha = np.mean(diags)
    _beta = np.mean(main_diag)
    _lambda = (np.sum(area) - np.sum(diags)) / (beats_count * beats_count - len(diags))

    rho_alpha = _alpha / _lambda
    rho_bera = _beta / _lambda

    return rho_alpha, rho_bera

def filter_1d(x, sdm, time_len = 48):
    diag_index = x[1] - x[0]
    diagonal = np.diag(sdm, -diag_index)

    if len(diagonal) < time_len:
        return -1

    window = np.ones(time_len)
    inside_sum = np.convolve(diagonal, window, mode = 'valid')
    diag_sum = np.sum(diagonal)
    outside_sum = diag_sum - inside_sum

    rate = (inside_sum / time_len) / (outside_sum / (len(diagonal) - time_len))

    return rate

def chorus_detection(filename, is_plot = False):
    # extract audio feature "/Users/xueweiyao/Downloads/musics/刘欢 - 得民心者得天下.mp3"
    audio = read_audio(filename)
    beats, beats_time = extract_beat(audio)
    mfcc = extract_mfcc_by_beat(audio, beats)
    chroma = extract_chroma_by_beat(audio, beats)

    # calculate the self-distance matrix
    sdm_chroma = calculate_sdm(chroma, is_normalization = False)
    sdm_mfcc = calculate_sdm(mfcc, is_normalization = False)



    #enhance the self-distance matrix
    enhanced_mat = enhance_sdm(sdm_chroma)

    # sum the mfcc self-distance matrix and enhanced chroma self-distance matrix
    sdm_new = enhanced_mat + sdm_mfcc


    bimar, indeces = detect_repetition(sdm_new)
    segments, bimar = locate_interesting_segment(bimar, indeces, beats_time)
    scores = calculate_segments_scores(sdm_new, bimar, segments, audio, beats)
    final_score, best = select_segment_most_likely_chorus(scores, beats_time, segments)

    #time = get_segment_time(best, beats_time)

    chorus = find_location_of_chorus(best, sdm_new)

    if is_plot:
        plt.matshow(sdm_chroma, cmap = plt.cm.gray)
        plt.matshow(sdm_mfcc, cmap = plt.cm.gray)
        plt.matshow(enhanced_mat, cmap = plt.cm.gray)
        plt.matshow(sdm_new, cmap = plt.cm.gray)
        plt.matshow(bimar, cmap = plt.cm.gray)

        plt.show()

    return chorus


