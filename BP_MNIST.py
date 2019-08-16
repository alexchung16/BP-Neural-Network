# -*- coding: utf-8 -*-
# @ fuction 利用BP网络训练 MNIST 数据集进行分类
# @ author alexchung
# @ date 13/8/2019 AM 10:52

import BP_Network_MSE
import BP_Network_Cross_Entropy
from MNIST_Loader import load_data_wrapper


if __name__ == "__main__":
    # 输入数据
    training_data, validation_data, testing_data = load_data_wrapper()
    # net = BP_Network_MSE.BPNetwork([784, 30, 10])
    net = BP_Network_Cross_Entropy.BPNetwork([784, 30, 10])
    net.SGD(training_data, 30, 10, 3, 0, testing_data)
