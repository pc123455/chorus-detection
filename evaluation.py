import numpy as np

def evaluate(annotation, chorus):
    # print annotation
    offset = 5
    anno_num = (len(annotation) - offset) / 2
    overlap_len = 0
    anno_len = 0
    chorus_len = 0
    chorus_num = len(chorus)
    for i in range(chorus_num):
        chorus_len += chorus[i, 1] - chorus[i, 0]

    for i in range(anno_num):
        anno = (float(annotation[i * 2 + offset]), float(annotation[i * 2 + 1 + offset]))
        for j in range(chorus_num):
            overlap_len += calculate_overlap(anno, chorus[j])
        anno_len += anno[1] - anno[0]

    recall = overlap_len / anno_len
    precision = overlap_len / chorus_len

    return recall, precision

def calculate_overlap(field_1, field_2):
    start_1 = field_1[0]
    end_1 = field_1[1]
    start_2 = field_2[0]
    end_2 = field_2[1]

    overlap_len = np.min([end_1, end_2]) - np.max([start_1, start_2])
    if overlap_len < 0:
        overlap_len = 0

    return overlap_len