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

def write_to_file(filename, content):
    """write content into file"""
    f = codecs.open(filename, "a", "utf-8")
    f.write(content)
    f.close()

def write_result_to_file(filename, recall, precision):
    """write the result into file"""
    recall = map(lambda num : str(num), recall)
    precision = map(lambda num: str(num), precision)

    recall_str = ','.join(recall) + '\n'
    precision_str = ','.join(precision) + '\n'
    write_to_file(filename, 'recall:' + recall_str)
    write_to_file(filename, 'precision:' + precision_str)

if __name__ == '__main__':
    path = '/Users/xueweiyao/Downloads/chorus/'
    content = read_from_file(path + 'annotation.csv')
    result_file = 'result.txt'

    recall = np.zeros(len(content))
    precision = np.zeros(len(content))
    f_measure = np.zeros(len(content))
    for i in range(len(content)):
        item = content[i]
        infos = item.split(',')
        music_name = infos[0]
        chorus = chorus_detection.chorus_detection(path + music_name, False)

        recall[i], precision[i] = evaluation.evaluate(infos, chorus)

    write_result_to_file(path + result_file, recall, precision)