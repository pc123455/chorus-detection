import numpy as np
import matplotlib.pyplot as plt

def quantify(matrix, depth = 8):
    image = matrix.copy()
    minimum = np.min(image)
    maximum = np.max(image)
    image = (image - minimum) / (maximum - minimum)
    image *= (pow(2, depth) - 1)
    image = image
    return image


def getGray(img, depth = 8):
    # numGray = [0 for i in range(pow(2, img.depth))]
    numGray = [0 for i in range(pow(2, depth))]
    height, width = img.shape
    for h in range(height):
        for w in range(width):
            numGray[int(round(img[h, w]))] += 1
    return numGray


def getThres(gray, depth = 8):
    maxV = 0
    bestTh = 0
    w = [0 for i in range(len(gray))]
    px = [0 for i in range(len(gray))]
    w[0] = gray[0]
    px[0] = 0
    for m in range(1, len(gray)):
        w[m] = w[m - 1] + gray[m]
        px[m] = px[m - 1] + gray[m] * m
    for th in range(len(gray)):
        w1 = w[th]
        w2 = w[len(gray) - 1] - w1
        if (w1 * w2 == 0):
            continue
        u1 = px[th] / w1
        u2 = (px[len(gray) - 1] - px[th]) / w2
        v = w1 * w2 * (u1 - u2) * (u1 - u2)
        if v > maxV:
            maxV = v
            bestTh = th
    return bestTh

def binarize(img, threshold, isplot = 0):
    new_img = img.copy()
    for i in range(0, len(img)):
        for j in range(0, len(img[i, :])):
            if img[i, j] < threshold:
                new_img[i, j] = 0
            else:
                new_img[i, j] = 1

    if isplot == 1:
        plt.matshow(new_img, cmap = plt.cm.gray)
        plt.show()

    return new_img