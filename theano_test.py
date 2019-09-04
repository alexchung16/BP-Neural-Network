#-*- coding: utf-8 -*-
# @ fuction theano test
# @ author alexchung
# @ date 2/9/2019 PM 17:29

import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import pp
from theano import In
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.sparse as sp
from theano import sparse


if __name__ == "__main__":

    # 标量(scalar)操作
    x = T.scalar('x')
    y = T.scalar('y')
    z = x+y
    # function方法描述
    # inputs 输入列表
    # outputs 输出
    sum_scalar = function([x, y], z)
    # pp(pretty print) 打印操作
    print(pp(z))
    print(sum_scalar(2, 3))

    # 矩阵(matrix)操作
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    sum_matrix = function([x, y], z)
    a = np.random.randint(0, 9, (2, 3))
    b = np.random.randint(0, 9, (2, 3))
    print(sum_matrix(a, b))

    # 向量(vector)操作
    x = T.vector('x')
    y = T.vector('y')
    z = x**2 + y**2 + 2*x*y
    power_vector = function([x, y], z)
    a = np.random.randint(0, 9, 3).tolist()
    b = np.random.randint(0, 9, 3).tolist()
    print(a)
    print(b)
    print(power_vector(a, b))

    # 同时执行多个操作
    x = T.dmatrix('x')
    # sigmoid(logistic)
    z_sig = 1/(1+T.exp(-x))
    # tanh
    z_tanh0 = T.tanh(x)
    z_tanh1 = (T.exp(x) - T.exp(-x))/(T.exp(x) + T.exp(-x))
    # tanh to sigmoid
    z_tanh2sig = (1+T.tanh(x/2))/2
    sig_tanh = function([x], [z_sig, z_tanh0, z_tanh1, z_tanh2sig])
    a = np.random.randint(-2, 2, (2, 2))
    print(a)
    print(sig_tanh(a)[0], '\n', sig_tanh(a)[1], '\n', sig_tanh(a)[2], '\n', sig_tanh(a)[3])

    # 为参数设置默认值
    x, y, w = T.dscalars('x', 'y', 'w')
    z = (x + y) * w
    f = function([x, In(y, value=1), In(w, value=2, name='w_default')], z)
    print(f(3))  # (3+1)*2
    print(f(3, 4))  # (3+4)*2
    print(f(3, 4, 3))  # (3+4)*3
    # 通过变量名称修改默认
    print(f(3, 4, w_default=5))  # (3+4)*5

    # 使用共享变量(shared variable)
    state = shared(0)
    # 判断输入是否为标量
    inc = T.iscalar('inc')
    # 如果是标量，执行状态更新
    accumulator = function([inc], state, updates=[(state, state + inc)])
    # 更新状态
    print(state.get_value())  # 0
    accumulator(1)
    print(state.get_value())  # 0+1
    # 重新设置状态
    state.set_value(-1)
    print(state.get_value())  # -1
    accumulator(1)
    print(state.get_value())  # 1+(-1)

    # 忽略共享值
    fn_of_state = state * 2 + inc
    foo = T.scalar(dtype=state.dtype)
    skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
    print( skip_shared(1, 3))
    print(state.get_value())

    # 使用随机数
    # 序列随机数生成器(sequence of random number generator, srng)
    srng = RandomStreams(seed=0)
    # 均匀分布(uniform distribution)
    rv_uniform = srng.uniform((2, 2))
    # 正态分布
    rv_normal = srng.normal((2, 2))
    uniform_func = function([], rv_uniform)
    # no_default_updates 控制随机数生成器的状态不受调用返回函数的影响
    normal_func = function([], rv_normal, no_default_updates=True)

    uv0 = uniform_func()
    uv1 = uniform_func()
    nv0 = normal_func()
    nv1 = normal_func()
    print(uv0, uv1)  # 结果不同
    print(nv0, nv1)  # 结果相同

    # 标量微分操作
    x = T.scalar('x')
    y = x**2
    grad_y = T.grad(y, x)
    scalar_grad_func = function([x], grad_y)
    print(pp(grad_y))
    print(scalar_grad_func(2))

    # 矩阵微分操作(sigmoid)
    x = T.dmatrix('x')
    # 将普通数值转换为scalar
    y = T.sum(1/(1 + T.exp(-x)))
    grad_s = T.grad(y, x)
    matrix_grad_func = function([x], grad_s)
    m = np.random.randint(0, 9, (2, 2))
    print(pp(grad_s))
    print(matrix_grad_func(m))

    # 雅可比阵(jacobian)
    # 计算输出相对于输入的一阶偏导数
    x = T.dvector('x')
    y = x ** 2
    J, updates = theano.scan(lambda i, y, x: T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
    jacobian_func = function([x], J, updates=updates)
    print(jacobian_func([4, 4]))

    # 海森阵(hessian)
    # 计算输出相对于输入的二阶偏导数
    x = T.dvector('x')
    y = x ** 2
    cost = y.sum()
    gy = T.grad(cost, x)
    H, updates = theano.scan(lambda i, gy, x: T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
    hessian_func = function([x], H, updates=updates)
    print(hessian_func([4, 4]))

    # sparse 稀疏操作
    # csc 压缩列  csr 压缩行
    x = sparse.csc_matrix(name='x', dtype='int32')
    data, indices, indptr, shape = sparse.csm_properties(x)
    # structure operation
    y = sparse.structured_add(x, 2)
    cac_func = function([x], y)
    # csc_matrix function
    a = sp.csr_matrix(np.asarray([[0, 1, 2], [0, 1, 0], [1, 0, 0]]))
    print(a.toarray())
    # csr to csc
    print(cac_func(a).toarray())




















