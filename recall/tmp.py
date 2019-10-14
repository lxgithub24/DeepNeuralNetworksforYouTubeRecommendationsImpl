# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 召回部分
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from recall.data_process import get_batch, get_total_num


class Args:
    video_total_num = 2
    search_total_num = 2
    dnn_depth = 3
    num_class = 1201
    learning_rate = 0.01
    batch_num = 10000
    epoch = 10
    checkpoint_dir = '../data/checkpoint_dir/'


class RecallModel:
    def __init__(self, args):
        self.video_total_num = args.video_total_num
        self.search_total_num = args.search_total_num
        self.depth = args.dnn_depth
        self.num_class = args.num_class
        self.num_samples = 100  # 负采样样本数
        self.units_list = [128] * self.depth
        self.learning_rate = args.learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.class_distinct = 1201
        self.batch_num = args.batch_num
        self.init_graph()

    def init_graph(self):
        # 初始化喂入参数，placeholder名字要唯一，不能更改placeholder的任何信息
        self.video_ids_ph = tf.placeholder(tf.int32, shape=[None, None], name='video_ids')
        self.search_id_ph = tf.placeholder(tf.int32, shape=[None], name='search_id')
        self.age_ph = tf.placeholder(tf.float32, shape=[None], name='age')
        self.gender_ph = tf.placeholder(tf.float32, shape=[None], name='gender')
        self.label_ph = tf.placeholder(tf.float32, shape=[None], name='label_ph')

        print(self.label_ph)
        self.label = self.label_ph
        # 很奇怪的问题，label必须二维的，但是biases却是一维的
        self.labels1 = tf.reshape(self.label, shape=[1, -1])
        print(self.labels1)

    def train(self, session, video_ids, search_id, age, gender, label):
        step = session.run([self.global_step],
                                    feed_dict={self.video_ids_ph: video_ids,
                                               self.search_id_ph: search_id,
                                               self.age_ph: age,
                                               self.gender_ph: gender,
                                               self.label_ph:label
                                               })
        return step


# 运行图
def run_graph():
    args = Args()
    # 可以增加gpu使用的设置
    with tf.Session() as session:
        recall_model = RecallModel(args)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        count = int(get_total_num() / args.batch_num) + 1
        for i in range(args.epoch):
            for j in range(count):
                has_more, video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = get_batch()
                print('#############')
                print(video_ids_batch)
                print(search_ids_batch)
                print(age_batch)
                print(gender_batch)
                print(label_batch)
                print('#############')
                if not has_more:
                    break
                step = recall_model.train(session, video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch)


if __name__ == '__main__':
    run_graph()
