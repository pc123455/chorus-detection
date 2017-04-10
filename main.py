import codecs
import chorus_detection
import evaluation
import numpy as np
import json
import multiprocessing

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

def test_param(islocal, min_sdm_window_size):
    for i in range(len(content)):
        item = content[i]
        infos = item.split(',')
        music_name = infos[0]
        print music_name
        chorus = chorus_detection.chorus_detection(path + music_name, min_sdm_window_size = min_sdm_window_size, is_local = islocal)
        recall[i], precision[i] = evaluation.evaluate(infos, chorus)

    return

def exp_for_one_param(param, musics):
    recall = np.zeros(len(musics))
    precision = np.zeros(len(musics))
    for i in range(len(musics)):
        item = content[i]
        infos = item.split(',')
        music_name = infos[0]
        print music_name
        chorus = chorus_detection.chorus_detection(path + music_name,\
                                                   min_sdm_window_size = param['min_sdm_window_size'],\
                                                   is_local = param['is_local'])
        recall[i], precision[i] = evaluation.evaluate(infos, chorus)

    return recall, precision


if __name__ == '__main__':
    path = '/Users/xueweiyao/Downloads/chorus/'
    content = read_from_file(path + 'annotation.csv')
    result_file = 'result.txt'
    data = []
    params = []

    params.append({'min_sdm_window_size': 16, 'is_local': True})
    params.append({'min_sdm_window_size': 32, 'is_local': True})
    params.append({'min_sdm_window_size': 48, 'is_local': True})
    params.append({'min_sdm_window_size': 16, 'is_local': False})

    # recall = np.zeros(len(content))
    # precision = np.zeros(len(content))
    # f_measure = np.zeros(len(content))
    # for i in range(len(content)):
    #     item = content[i]
    #     infos = item.split(',')
    #     music_name = infos[0]
    #     print music_name
    #     chorus = chorus_detection.chorus_detection(path + music_name, min_sdm_window_size = 48, is_local = True)
    #     recall[i], precision[i] = evaluation.evaluate(infos, chorus)

    for param in params:
        recall, precision = exp_for_one_param(param, content)
        # multi processing
        data.append({'recall': recall, 'precision': precision, 'is_local': param['is_local'], 'min_sdm_window_size': 48})

    data = {'data': data}
    data = json.dump(data)
    write_to_file(result_file, data)
    # write_result_to_file(result_file, recall, precision)