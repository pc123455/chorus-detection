import numpy as np

def quantify(matrix, depth = 8):
    black = 0
    white = pow(2, depth) - 1
    image = matrix.copy()
    maximum = np.max(matrix)
    coeff = float(white) / maximum
    image *= coeff
    return image


def getGray(img, depth = 8):
    # numGray = [0 for i in range(pow(2, img.depth))]
    numGray = [0 for i in range(pow(2, depth))]
    height, width = img.shape
    for h in range(height):
        for w in range(width):
            numGray[int(img[h, w])] += 1
    return numGray


def getThres(gray):
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