import numpy as np

def evaluate(annotation, chorus):
    anno_num = (len(annotation) - 2) / 2
    overlap_len = 0
    anno_len = 0
    chorus_len = chorus[1] - chorus[0]
    for i in range(1, anno_num + 1):
        anno = (float(annotation[i * 2]), float(annotation[i * 2 + 1]))
        overlap_len += calculate_overlap(anno, chorus)
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