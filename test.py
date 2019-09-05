#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File test.py
# @ Description 测试
# @ Author alexchung
# @ Time 17/7/2019 PM 13:56


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


DATA_PATH = 'F:/python_code/datasets'


def load_csv_data(data_path=DATA_PATH, data=''):
    """
    加载csv文件
    :param data_path: 数据路径
    :param data: csv表名
    :return:
    """
    csv_path = os.path.join(data_path, data)
    return pd.read_csv(csv_path, header=None)


class enterprise:
    def __init__(self, num, price):

        self.num = num
        self.price = price
    def cost(self):
        return self.num*self.price


if __name__ == "__main__":
    # wine = load_csv_data(data='winequality-white.csv')
    #
    # # array_data = data.as_matrix()
    # # print('alex')
    #
    # # test zip
    # # zip() 函数用于将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
    # ls = [1, 2, 3]
    # lb = [4, 5, 6]
    # data = [np.random.randn(b, 1) for b in ls[1:]]
    # zlsb = zip(ls, lb)
    # for a, b in zlsb:
    #     print(a, b)
    # print(list(zlsb))
    #
    # # test argmax
    # # argmax返回的是最大数的索引
    # print(np.argmax(wine.values, axis=0))
    # np.random.seed(20190726)
    # rdn = np.random.randn(3, 4)
    # rdint = np.random.randint(0, 10, size=(3, 4))
    # rdint0 = np.random.randint(0, 10, size=(3, 1))
    # args = np.argmax(rdn, axis=0)
    # print(rdn, args)
    #
    # # test sigmoid
    # def sigmoid(z):
    #     """
    #     sigmod函数
    #     :param z: 普通函数值
    #     :return: sigmod 变换函数值
    #     """
    #     return 1.0 / (1.0 + np.exp(-z))
    #
    # print(rdint[:, 0])
    # print(sigmoid(rdint[:, 0]))
    #
    # # test transpose
    # # 转置
    # p = np.arange(4).reshape(2, 2)
    # pt = p.transpose()
    # p1 = np.arange(24).reshape(2, 3, 4)
    # print(p1)
    # # 不带参数全转置 2*3*4 -> 4*3*2
    # pt1 = p1.transpose()
    # print(pt1)
    # # 带参数 z 对第二三维进行转置 2*3*4->2*4*3
    # pt2 = p1.transpose(0, 2, 1)
    # print(pt2)
    #
    # # 获取路径
    # # print(os.getcwd())
    # # print(os.path.abspath('.'))
    # # print(os.path.abspath('..'))
    #
    # # test @property @classmethod @staticmethod
    # class cal:
    #     cal_name = '计算器'
    #
    #     def __init__(self, x, y):
    #         self.x = x
    #         self.y = y
    #
    #     @property           # 在cal_add函数前加上@property，使得该函数可直接调用，封装起来
    #     def cal_add(self):
    #         return self.x + self.y
    #
    #     @classmethod        # 在cal_info函数前加上@classmethon，则该函数变为类方法，该函数只能访问到类的数据属性，不能获取实例的数据属性
    #     def cal_info(cls):  # python自动传入位置参数cls就是类本身
    #         print('这是一个{0}'.format(cls.cal_name))   # cls.cal_name调用类自己的数据属性
    #
    #     @staticmethod       # 静态方法 类或实例均可调用
    #     def cal_test(a, b, c):  # 改静态方法函数里不传入self 或 cls
    #         print(a, b, c)
    # # c1 = cal(10, 11)
    # # c1.cal_info()
    # # c1.cal_test(1, 2, 3)
    # # print(c1.cal_add)
    #
    #
    # # randint test
    # p = np.random.RandomState(0).randint(20)
    # print(p)

    # 读取文件夹文件信息，转换名称为数值标签
    path = 'F:\\databases\\fruit_vegetables'
    contents = os.listdir(path)
    classes = [x for x in contents if os.path.isdir(os.path.join(path, x))]
    apple_path = os.path.join(path, 'apple')
    apple_list = os.listdir(apple_path)
    labels = []
    images = []
    for ii, file in enumerate(apple_list, 1):
        images.append(os.path.join(path, 'apple', file))
        labels.append(1)
    lf = LabelEncoder().fit(labels)
    dataLabel = lf.transform(labels).tolist()
    print(dataLabel)









