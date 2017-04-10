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
