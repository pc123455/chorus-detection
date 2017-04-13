import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    size = 32
    filter = np.ones([size, size])
    for i in range(size / 2):
        filter[i, size / 2 + i] = 0
        filter[size / 2 + i, i] = 0

    for i in range(size):
        filter[i, i] = 0

    plt.matshow(filter, cmap = plt.cm.gray)
    plt.title('2D filter whose size is 32')
    plt.show()


data = {'data': [{'is_local': True, 'recall': [ 0.4375    ,  0.54716981,  0.27906977,  0.23529412,  0.33050847,
        0.19230769,  0.02439024,  0.73076923,  0.19354839,  0.5       ,
        0.46875   ,  0.16129032,  0.        ,  0.45054945,  0.04651163,
        0.64130435,  0.11764706,  0.41025641], 'precision': [ 0.4375    ,  0.54716981,  0.27906977,  0.23529412,  0.33050847,
        0.19230769,  0.02439024,  0.73076923,  0.19354839,  0.5       ,
        0.46875   ,  0.16129032,  0.        ,  0.45054945,  0.04651163,
        0.64130435,  0.11764706,  0.41025641], 'min_sdm_window_size': 48}]}