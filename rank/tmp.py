# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 17:25
# @Author  : RIO
# @desc: TODO:DESC
import tensorflow as tf

input = tf.constant([[0,1,2],[3,4,5]])

tf.Session().run(tf.local_variables_initializer())
print(input.shape[0])
print(input.get_shape())
print(tf.shape(input))

