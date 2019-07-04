# -*- coding: utf-8 -*-

# @Author: zhoujiamu
# @Time:  2019/3/24 上午12:15

"""
 多层全连接神经网络
"""

import tensorflow as tf
from numpy.random.mtrand import RandomState


def get_weight(shape, lambda_):
    """
    获取一层神经网络边上的权重，并将权重的L2正则损失加入名称为loses的集合中
    :param shape: 权重维度
    :param lambda_: 正则项系数
    :return: 权重变量
    """
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

    # 把该权重的L2正则加入到loses集合
    tf.add_to_collection('loses', tf.contrib.layers.l2_regularizer(lambda_)(var))

    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 128

# 定义每一层网络的节点的个数
layer_dim = [2, 64, 128, 128, 64, 64, 1]

# 神经网络的层数
n_layers = len(layer_dim)

# 当前层输入, 初始时候为输入层
current_layer = x

# 当前层节点个数, 初始时候为输入层节点个数
current_dim = layer_dim[0]

# 通过循环生成 layer_dim 层全连接的神经网络结构
for i in range(1, n_layers):
    next_dim = layer_dim[i]     # 下一层节点个数
    weight = get_weight([current_dim, next_dim], 0.001)     # 生成当前层权重，并把L2正则加入计算图集合
    bias = tf.Variable(tf.constant(0.1, shape=[next_dim]))

    # 使用 ReLU 激活函数
    current_layer = tf.nn.relu(tf.matmul(current_layer, weight) + bias)

    # 进入下一层前将当前层节点个数更新为下一层输入节点个数
    current_dim = next_dim

# 定义数据拟合损失部分
mse_loss = tf.reduce_mean(tf.square(y_ - current_layer))

# 将数据拟合损失部分加入计算图集合
tf.add_to_collection('loses', mse_loss)

# 使用 get_collection 方法返回所有计算图集合中的损失并加起来得到总损失

loss = tf.add_n(tf.get_collection('loses'))

# 定义优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)

dataset_size = 1000000

X = rdm.rand(dataset_size, 2)

Y = [[x1 + x2] for (x1, x2) in X]

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 设定训练轮数
    STEPS = 10000

    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(mse_loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), mse_loss on all data is %g" % (i, total_cross_entropy))

            total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), all_loss on all data is %g" % (i, total_cross_entropy))

    total_cross_entropy = sess.run(mse_loss, feed_dict={x: X, y_: Y})
    print("After %d training step(s), mse_loss on all data is %g" % (i, total_cross_entropy))

    total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
    print("After %d training step(s), all_loss on all data is %g" % (i, total_cross_entropy))

