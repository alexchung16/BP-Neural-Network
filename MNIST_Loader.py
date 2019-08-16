#-*- coding: utf-8 -*-
# @ fuction 手撕BP神经网络
# @ author alexchung
# @ date 12/8/2019 AM 09:27

from __future__ import print_function
import os
import _pickle as cPickle
import gzip
import numpy as np


def load_data():
    """
    加载 mnist 数据
    :return: 训练数据 验证数据 测试数据
    """
    f = gzip.open(os.getcwd()+'/mnist/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    数据集格式转换
    :return:
    """
    train_data, validate_data, test_data = load_data()
    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_target = [vectorised_result(y) for y in train_data[1]]

    training_data = zip(train_inputs, train_target)
    # 验证集
    validation_inputs = [np.reshape(v, (784, 1)) for v in validate_data[0]]
    validation_data = zip(validation_inputs, validate_data[1])
    # 测试机
    test_inputs = [np.reshape(v, (784, 1)) for v in test_data[0]]
    testing_data = zip(test_inputs, test_data[1])

    return training_data, validation_data, testing_data


def vectorised_result(y):
    """
    格式化标签为10*1维array
    :param y:
    :return:
    """
    vector_y = np.zeros(shape=(10, 1))
    vector_y[y] = 1
    return vector_y


if __name__ == "__main__":
    train_data, validation_data, testing_data = load_data_wrapper()
    print(list(train_data)[1])

