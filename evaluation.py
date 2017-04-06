import numpy as np

def read_annotation(filename):
    input = open(filename, 'r')

def calculate_overlap(field_1, field_2):
    start_1 = field_1[0]
    end_1 = field_1[1]
    start_2 = field_2[0]
    end_2 = field_2[0]

    overlap_len = np.min([end_1, end_2]) - np.max([start_1, start_2])
    if overlap_len < 0:
        overlap_len = 0

    return overlap_len