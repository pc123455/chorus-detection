import codecs
import chorus_detection
import evaluation
import numpy as np

def read_from_file(filename):
    f = codecs.open(filename, 'r')
    line = f.readline()
    list = []
    while line:
        list.append(line)
        line = f.readline()
    f.close()
    return list

if __name__ == '__main__':
    path = '/Users/xueweiyao/Downloads/chorus/'
    content = read_from_file(path + 'annotation.csv')

    recall = np.zeros(len(content))
    precision = np.zeros(len(content))
    f_measure = np.zeros(len(content))
    for i in range(len(content)):
        item = content[i]
        infos = item.split(',')
        music_name = infos[0]
        chorus = chorus_detection.chorus_detection(path + music_name, False)

        recall[i], precision[i] = evaluation.evaluate(infos, chorus)

        print recall[i]
        print precision[i]
        print chorus

    print recall
    print precision