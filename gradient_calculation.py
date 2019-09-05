#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File gradient_calculation.py
# @ Description 梯度近似计算
# @ Author alexchung
# @ Time 27/8/2019 PM 20:00


import numpy as np


def func(x):
    """
    平方函数
    :param x:
    :return:
    """
    return np.sum(np.power(x, 2))


def numerical_gradient(f, x):
    """
    梯度数值
    :param f: 多元函数
    :param x: 坐标点
    :return:
    """
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        # 获取并存储某个维度坐标
        tmp_val = x[idx]

        # 更新坐标，求左点
        x[idx] = tmp_val - h

        f_left = f(x)
        # 更新坐标，求右点
        x[idx] = tmp_val + h
        f_right = f(x)

        # 求取近似偏导 并保存
        grad[idx] = (f_right - f_left) / (2*h)

        # 恢复维度坐标值
        x[idx] = tmp_val
    return grad


if __name__ == "__main__":
    x = [3., 4.]
    grad = numerical_gradient(func, np.array(x))
    print(grad)




