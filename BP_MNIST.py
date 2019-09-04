# -*- coding: utf-8 -*-
# @ fuction 利用BP网络训练 MNIST 数据集进行分类
# @ author alexchung
# @ date 13/8/2019 AM 10:52

import numpy as np
import random
import matplotlib.pyplot as plt

from MNIST_Loader import load_data_wrapper, load_data_shared
import CNN_Network
from CNN_Network import CNNNetwork
from CNN_Network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

def readTrainingData(train_data):
    """
    zip 格式 train_data
    :param train_data:
    :return:
    """

    i = 0
    training_data = list(train_data)
    train_image = np.zeros(((training_data[0][0]).shape[0], len(training_data)))
    train_label = np.zeros(((training_data[0][1]).shape[0], len(training_data)))

    for image, label in training_data:
        train_image[:, i] = image.reshape((training_data[0][0]).shape[0],)
        train_label[:, i] = label.reshape((training_data[0][1]).shape[0],)
        i += 1
    return train_image, train_label


def showImage(img_data):
    """
    图像显示函数
    :param img_data: 图像数据
    :return:
    """
    plt.imshow(img_data, cmap=plt.cm.binary)
    plt.show()


if __name__ == "__main__":

    # 测试一 全连接网络(full connect)
    # # 输入数据
    # training_data, validation_data, testing_data = load_data_wrapper()
    # # 提取zip格式数据de
    #
    # training_data = list(training_data)
    # validation_data = list(validation_data)
    # testing_data = list(testing_data)
    # # net = BP_Network_MSE.BPNetwork([784, 30, 10]
    # net = BP_Network_Cross_Entropy.BPNetwork([784, 30, 30, 10])
    # net.optimizationWeightInitialization()
    # net.SGD(training_data, 30, 10, 0.1, 5.0, validation_data,
    #         monitor_training_cost=True, monitor_training_accuracy=True,
    #         monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
    #         )

    # 测试二 卷积神经网路测试(CNN)
    training_data, validation_data, test_data = load_data_shared()
    mini_batch_size = 10
    net = CNNNetwork([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      pool_size=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)],
        mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)


    # 测试三 读取并显示训练数据
    # train_image, train_label = readTrainingData(training_data)
    # showImage(train_image[:, 1]
    # .reshape(28, 28))



