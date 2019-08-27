#-*- coding: utf-8 -*-
# @ fuction 手撕BP神经网络(基于交叉损失熵 Cross Entropy)
# @ author alexchung
# @ date 25/7/2019 PM 16:46

import sys
import numpy as np
import random


class MeanSquareCost(object):

    @staticmethod
    def fn(a, y):
        """
        输出激活输出量和期望值的交叉损失熵
        :param a: 激活输出量
        :param y: 期望值
        :return:
        """
        # np.linalg.norm 求取矩阵或向量范数（Matrix or vector norm.），默认2范数
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """
        返回输出层的误差(delta)
        :param z:
        :param a:
        :param y:
        :return:
        """
        return(a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        输出激活函数值和期望值的交叉损失熵
        :param a: 激活函数值
        :param y: 期望值
        :return:
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        返回输出层的误差(delta)
        :param z:
        :param a:
        :param y:
        :return:
        """
        return a-y


class BPNetwork(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        :param sizes: 网络中各层的神经元的数量
        """
        # 获取网络层数和各层神经元数
        self.sizes = sizes
        self.cost = cost
        # 获取网络层数
        self.num_layers = len(sizes)
        self.optimizationWeightInitialization()

    def optimizationWeightInitialization(self):
        """
        改进权重初始化
        均值 = 0
        标准差 = 1/sqrt(n)
        :return:
        """
        # 初始化隐含层和输出层神经元偏置矩阵
        self.biases = [np.random.randn(b, 1) for b in self.sizes[1:]]
        # 初始化隐含层和输出层神经元的权重矩阵
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def rawWeightInitialization(self):
        """
        原始权重初始化
        均值 = 0
        标准差 = 1
        :return:
        """
        # 初始化隐含层和输出层神经元偏置矩阵
        self.biases = [np.random.randn(b, 1) for b in self.sizes[1:]]
        # 初始化隐含层和输出层神经元的权重矩阵
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, x):
        """
        前馈操作
        :param x: 单个训练样本
        :return:
        """
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta,  lmbda=0.0, evaluation_data=None,
            monitor_training_cost=False, monitor_training_accuracy=False,
            monitor_evaluation_cost=False, monitor_evaluation_accuracy=False):
        """
        随机梯度下降（Stochastic Gradient Descent)
        :param train_data: 训练数据元祖(x,y)列表
        :param epochs: 迭代轮数
        :param mini_batch_size: 最小批大小
        :param eta: 学习率
        :param lmbda(lambda): 正则化参数
        :param evaluation_data:  评估数据集
        :param monitor_evaluation_cost: 监控评估数据损失
        :param monitor_evaluation_accuracy: 监控评估数据准确性
        :param monitor_train_cost: 监控训练数据损失
        :param monitor_train_accuracy: 监控训练数据准确性
        :return:
        """

        # 获取 zip 对象存储数据
        train_data = list(training_data)
        eval_data = list(evaluation_data)
        # 获取训练数据和评估数据的大小
        n_train = len(train_data)
        n_eval = 0
        if evaluation_data:
            n_eval = len(eval_data)
        # 初始化监控参数列表
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for r in range(epochs):
            random.shuffle(train_data)
            # 获取最小批
            mini_batches = [
                train_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # 利用最小批更新权重和偏置
                self.update_mini_batch(mini_batch, eta, lmbda, len(train_data))

            # 输出当前迭代次数
            print("Epoch {0} training complete".format(r))
            if monitor_training_cost:
                cost = self.total_cost(train_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(train_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n_train))
            if monitor_evaluation_cost:
                cost = self.total_cost(eval_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(eval_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(eval_data), n_eval))

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        应用梯度下降更新最小批
        *++++
        :param mini_batch: 最小批
        :param eta: 学习率
        :param lmbda: 正则化参数
        :param lmbda: 训练数据集大小
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

        
        # L1正则化
        # self.weights = [w - eta*(lmbda/n)*sgn(w) - (eta * nw) / len(mini_batch) for w, nw in
        #                 zip(self.weights, nabla_w)]
        # L2正则化
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta * nw) / len(mini_batch) for w, nw in
                        zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb) / len(mini_batch) for b, nb in zip(self.biases, nabla_b)]

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
        # 关键区别（唯一区别）
        ######################################################
        # 均方误差损失
        # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # 交叉熵损失
        delta = self.cost.delta(zs[-1], activations[-1], y)
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

    def accuracy(self, data, convert=False):
        """
         评估函数
        :param data: 评估数据
        :param convert: 是否执行转换
        :return:
        """
        # 当数据集为 training data 标签 shape = (10,1) 为 onehot(独热编码) 需要解码
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        # 当数据集为 validation data 或 test data 标签 shape = (1,1) 不需要解码
        else:
            results = [(np.argmax(self.feedforward(x)), y) for x, y in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        损失函数
        :param data:
        :param labda:
        :param convert:
        :return:
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        # 正则化项
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def cost_derivative(self, out_activations, y):
        """
        计算激活输出量的偏导
        :param out_activations: 激活输出量
        :param y: 标签
        :return:
        """
        return out_activations - y


def vectorized_result(j):
    """
    生成独热label
    :param j: 十进制数值
    :return:
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

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
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


def sgn(x):
    """
    sgn函数
    :param x:
    :return:
    """
    sgn = lambda x: 1 if x > 0 else -1
    # x = [sgn(x[i][j]) for i, j in zip(range(x.shape[0]), range(x.shape[1]))]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = sgn(x[i][j])
    return x


if __name__ == "__main__":

    a = np.random.rand(2, 2)
    print(a)
    b = sgn(a)
    print(b)




# 测试结果
# parameter
# net = BP_Network_Cross_Entropy.BPNetwork([784, 100, 10])
# net.SGD(training_data, 30, 10, 0.5, 0.0, testing_data)

# Epoch 0:9287/10000
# Epoch 1:9407/10000
# Epoch 2:9468/10000
# Epoch 3:9554/10000
# Epoch 4:9546/10000
# Epoch 5:9567/10000
# Epoch 6:9584/10000
# Epoch 7:9592/10000
# Epoch 8:9602/10000
# Epoch 9:9560/10000
# Epoch 10:9628/10000
# Epoch 11:9632/10000
# Epoch 12:9629/10000
# Epoch 13:9625/10000
# Epoch 14:9624/10000
# Epoch 15:9622/10000
# Epoch 16:9648/10000
# Epoch 17:9658/10000
# Epoch 18:9639/10000
# Epoch 19:9667/10000
# Epoch 20:9659/10000
# Epoch 21:9638/10000
# Epoch 22:9652/10000
# Epoch 23:9648/10000
# Epoch 24:9663/10000
# Epoch 25:9659/10000
# Epoch 26:9639/10000
# Epoch 27:9660/10000
# Epoch 28:9649/10000
# Epoch 29:9664/10000

