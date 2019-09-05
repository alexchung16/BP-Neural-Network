#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File BP_Network_MSE.py
# @ Description 手撕BP神经网络(基于均方误差损失 Mean Square Error)
# @ Author alexchung
# @ Time 25/7/2019 PM 16:46

import os
import numpy as np
import random


class BPNetwork(object):
    def __init__(self, sizes):
        """
        :param sizes: 网络中各层的神经元的数量
        """
        # 获取网络层数和各层神经元数
        self.sises = sizes
        # 获取网络层数
        self.num_layers = len(sizes)
        # 初始化隐含层和输出层神经元偏置矩阵
        self.biases = [np.random.randn(b, 1) for b in sizes[1:]]
        # 初始化隐含层和输出层神经元的权重矩阵
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        """
        前馈操作
        :param x: 单个训练样本
        :return:
        """
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """
        随机梯度下降（Stochastic Gradient Descent)
        :param train_data: 训练数据元祖(x,y)列表
        :param epochs: 迭代轮数
        :param mini_batch_size: 最小批大小
        :param eta: 学习率
        :param test_data: 测试数据
        :return:
        """
        # 获取 zip 对象存储数据
        train_data = list(train_data)
        test_data = list(test_data)
        # 获取测试数据大小
        n_train = len(train_data)
        n_test = 0
        if test_data:
            n_test = len(test_data)

        for r in range(epochs):
            random.shuffle(train_data)
            # 获取最小批
            mini_batches = [
                train_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # 利用最小批更新权重和偏置
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(r, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(r))


    def update_mini_batch(self, mini_batch, eta):
        """
        应用梯度下降更新最小批
        *++++
        :param mini_batch: 最小批
        :param eta: 学习率
        :return:
        """
        # 初始化存储权重和偏差的微分算子(nabla)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 获取损失函数与权重和偏差的梯度
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            # 求取损失函数批次梯度之和
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 利用最小批权重和偏置梯度平均值更新权重和偏差
        self.weights = [w-(eta*nw)/len(mini_batch) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta*nb)/len(mini_batch) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        利用反向传播(backprogation)算法计算梯度
        :param x: 数据
        :param y: 标签
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 计算正向传播激活函数值(feed forward)
        activation = x
        activations = [x]  # 存储各层的激活函数值

        zs = []  # 存储各层的z向量（激活前数值）, 计算激活函数导数

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 利用反向传播计算权重和偏置的梯度
        # 计算输出层的梯度, 并存储
        # 均方误差损失
        ######################################################
        # 均方误差损失
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # 交叉熵损失
        # delta = self.cost_derivative(activations[-1], y)
        ######################################################

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        # 计算隐藏层的梯度，并存储
        for l in range(2, self.num_layers):
            # 从L-1层反向计算
            z = zs[-l]
            sp = sigmoid_prime(z)
            # delta = np.dot(self.weights[-l+1], delta) * self.sigmoid_prime(zs[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        """
        评估函数
        :param test_data: 测试集
        :return:
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, out_activations, y):
        """
        计算激活输出量的偏导
        :param out_activations: 激活输出量
        :param y: 标签
        :return:
        """
        return out_activations - y


# 其他函数(miscellaneous)
def sigmoid(z):
    """
    sigmod函数
    :param z: 加权输出值
    :return: sigmod 变换函数值
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """
    sigmod函数导数
    :param z: 加权输出值
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


# 测试结果
# parameter
# training_data, validation_data, testing_data = load_data_wrapper()
# net = BP_Network_MSE.BPNetwork([784, 30, 10])
# net.SGD(training_data, 30, 10, 3, testing_data)

# Epoch 0:9010/10000
# Epoch 1:9202/10000
# Epoch 2:9278/10000
# Epoch 3:9367/10000
# Epoch 4:9362/10000
# Epoch 5:9402/10000
# Epoch 6:9372/10000
# Epoch 7:9415/10000
# Epoch 8:9450/10000
# Epoch 9:9454/10000
# Epoch 10:9476/10000
# Epoch 11:9472/10000
# Epoch 12:9482/10000
# Epoch 13:9488/10000
# Epoch 14:9463/10000
# Epoch 15:9462/10000
# Epoch 16:9496/10000
# Epoch 17:9495/10000
# Epoch 18:9484/10000
# Epoch 19:9452/10000
# Epoch 20:9495/10000
# Epoch 21:9500/10000
# Epoch 22:9462/10000
# Epoch 23:9502/10000
# Epoch 24:9465/10000
# Epoch 25:9506/10000
# Epoch 26:9494/10000
# Epoch 27:9494/10000
# Epoch 28:9499/10000
# Epoch 29:9499/10000