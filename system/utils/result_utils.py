import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    # 测试精度
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
    # 最大精度列表
    max_accurancy = []
    # 做times次，每次找对应times下最大的测试精度
    for i in range(times):
        max_accurancy.append(test_acc[i].max())
    # 求times次精度的std和mean
    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    # 测试精度
    test_acc = []
    # 长度为times的list，每个元素为算法名
    algorithms_list = [algorithm] * times
    # 循环记录每次的file_name，打开获得rs_test_acc，然后删除对应文件
    # 将rs_test_acc转换为np.array，然后添加到test_acc里
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    # 定义文件路径，即文件名
    file_path = "../results/" + file_name + ".h5"
    # 读取文件，获得rs_test_acc内容，转换为np.array
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
    # 输出rs_test_acc的长度，返回内容
    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc