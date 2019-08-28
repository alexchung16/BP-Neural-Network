# -*- coding: utf-8 -*-
# @ fuction 利用BP网络训练 MNIST 数据集进行分类
# @ author alexchung
# @ date 13/8/2019 AM 10:52

import numpy as np
import random
import matplotlib.pyplot as plt
import BP_Network_MSE
import BP_Network_Cross_Entropy
from MNIST_Loader import load_data_wrapper


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
    # 输入数据
    training_data, validation_data, testing_data = load_data_wrapper()

    # net = BP_Network_MSE.BPNetwork([784, 30, 10]
    net = BP_Network_Cross_Entropy.BPNetwork([784, 30, 10])
    net.optimizationWeightInitialization()
    # net.SGD(training_data, 30, 10, 0.1, 5.0, validation_data,
    #         monitor_training_cost=True, monitor_training_accuracy=True,
    #         monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
    #         )

    # train_image, train_label = readTrainingData(training_data)
    # showImage(train_image[:, 1].reshape(28, 28))



