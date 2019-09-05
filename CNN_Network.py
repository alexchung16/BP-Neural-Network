#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File CNN_Network.py
# @ Description 卷积神经网络
# @ Author alexchung
# @ Time 29/8/2019 PM 17:29

# 标准库
import os
import _pickle
import gzip

# 第三方库
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# 损失函数
# sigmoid
from theano.tensor.nnet import sigmoid
# 定义激活函数
# tanh
from theano.tensor import tanh
# linear 线性激活函数
linear = lambda z: z
# relu 修正线性单元
ReLU = lambda z: T.maximum(0.0, z)


class CNNNetwork(object):
    """
    CNN网络主类
    """
    def __init__(self, layers, mini_batch_size):
        """
        CNN 构造函数
        :param layers: 所有层参数信息列表
        :param mini_batch_size: 最小批大小
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        # 多层循环取各层参数
        self.params = [param for layer in self.layers for param in layer.params]
        # 符号变量
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        # input layers
        init_layer = layers[0]
        init_layer.set_input(self.x, self.x, self.mini_batch_size)
        # hider layers
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_input(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        # output layers(softmax)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        """
        最小批随机梯度训练数据
        :param train_data: 训练数据
        :param epoch: 迭代次数
        :param mini_batch_size: 最小批大小
        :param eta: 学习率
        :param validation_data: 评估数据
        :param test_data: 测试数据
        :param lmbda: 正则化参数
        :return:
        """

        # 获取数据
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # 计算最小批数量
        num_training_batches = int(size(training_data) / mini_batch_size)
        num_validation_batches = int(size(validation_data) / mini_batch_size)
        num_test_batches = int(size(test_data) / mini_batch_size)

        # 定义正则化损失函数 梯度符号 更新公式
        # L2 正则化权重平方项
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        # 损失函数
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        # theano 自动求梯度
        grads = T.grad(cost, self.params)
        # 更新权重
        updates = [(param, param - eta*grad) for param, grad in zip(self.params, grads)]

        # 定义函数训练最小批
        # 评估集(validation set)和测试集(test set)的最小批准确性
        i = T.lscalar()
        # 训练方法
        train_mb_func = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                    training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        # 评估集最小批准确性
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                    self.x:
                        validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                    self.y:
                        validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # 测试集最小批准确性
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # 测试集预测结果
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                train_mb_func(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class ConvPoolLayer(object):
    """
    定义卷积层
    """
    def __init__(self, filter_shape, image_shape, pool_size=(2, 2), activition_fn=sigmoid):
        """
        :param filter_shape: 卷积核形状（四元组）：卷积核数量 输入图像的特征映射的量 卷积核高 卷积核宽
        :param image_shape: 输入图像形状（四元组）：最小批， 输入图像的特征映射的量 图像高 图像宽
        :param activition_fn:
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn = activition_fn
        # 初始化权重核偏置
        # 获取权重(weight)数量, 用于初始化权重
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                                         dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                                          dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, inputs, input_dropout, mini_batch_size):
        """
        设置
        :param inputs: 其他正则化方法（除dropout）
        :param input_dropout: dropout 正则化
        :param mini_batch_size: 最小批大小
        :return:
        """
        self.inputs = inputs.reshape(self.image_shape)
        # 初始化卷积层
        conv_out = conv2d(input=self.inputs, filters=self.w, filter_shape=self.filter_shape, input_shape=self.image_shape)
        # 初始化池化层
        pool_out = pool.pool_2d(input=conv_out, ws=self.pool_size, ignore_border=True)
        # 获取激活输出量
        self.output = self.activation_fn(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # 卷积层无dropout
        self.output_dropout = self.output


class FullyConnectedLayer(object):
    """
    定义全连接层
    """
    def __init__(self, n_in, n_out, activition_fn=sigmoid, p_dropout=0.0):
        """
        构造函数
        :param n_in: 输入神经元数量
        :param n_out: 输出神经元数量
        :param activition_fn: 激活函数
        :param p_dropout:
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activition_fn
        self.p_dropout = p_dropout
        # 初始化权重核偏置
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1/n_out), size=(n_in, n_out)), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=theano.config.floatX),
            name='b', borrow=True
        )
        self.params = [self.w, self.b]

    def set_input(self, inputs, input_dropout, mini_batch_size):
        """
        设置
        :param inputs: 其他正则化方法（除dropout）
        :param input_dropout: dropout 正则化
        :param mini_batch_size: 最小批大小
        :return:
        """
        # 初始化输入(非dropout正则化)
        self.inputs = inputs.reshape((mini_batch_size, self.n_in))
        # 获取激活输出量
        self.output = self.activation_fn((1-self.p_dropout)*T.dot(self.inputs, self.w) + self.b)
        # 获取预测最大概率
        self.y_out = T.argmax(self.output, axis=1)

        # 初始化输入(dropout)
        self.input_dropout = dropout_layer(input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inputs, self.w) + self.b)

    def accuracy(self, y):
        """
        返回最小批准确率
        :param self:
        :param y: 标签
        :return:
        """
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer(object):
    """
    定义全连接层
    """
    def __init__(self, n_in, n_out, p_dropout=0.0):
        """
        构造函数
        :param n_in: 输入神经元数量
        :param n_out: 输出神经元数量
        :param p_dropout:
        """
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # 初始化权重核偏置
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, inputs, input_dropout, mini_batch_size):
        """
        设置
        :param inputs: 其他正则化方法（除dropout）
        :param input_dropout: dropout 正则化
        :param mini_batch_size: 最小批大小
        :return:
        """
        # 初始化输入(非dropout正则化)
        self.inputs = inputs.reshape((mini_batch_size, self.n_in))

        # 获取softmax输出
        self.output = softmax((1-self.p_dropout)*T.dot(self.inputs, self.w) + self.b)
        # 获取预测最大概率
        self.y_out = T.argmax(self.output, axis=1)

        # 初始化输入(dropout)
        self.input_dropout = dropout_layer(input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inputs, self.w) + self.b)

    def cost(self, net):
        """
        返回对数极大似然损失(log-likelihood cost)
        :param net:
        :return:
        """
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        """
        返回最小批准确率
        :param self:
        :param y: 标签
        :return:
        """
        return T.mean(T.eq(y, self.y_out))


# 其他函数
def size(data):
    """
    返回数据集合的大小
    :param data:
    :return:
    """
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    # 获取999999之间的一个随机数
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    # 获取掩模
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)







