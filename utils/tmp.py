# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 15:03
# @Author  : RIO
# @desc: TODO:DESC
import tensorflow as tf

a = tf.Variable([1,1])
print(a.shape)
session = tf.Session()
session.run(tf.global_variables_initializer())
b = session.run([a])
print(b)
