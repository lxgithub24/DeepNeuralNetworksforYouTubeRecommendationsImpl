# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 召回部分
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from recall.data_process import get_batch
from utils.utils import list2array
from utils.config import SuperPrams


class Args(SuperPrams):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.dnn_depth = 3
        self.learning_rate = 0.01
        self.epoch = 10
        self.checkpoint_dir = '../data/recall/checkpoint_dir/'


class RecallModel:
    def __init__(self, args):
        self.video_total_num = args.video_total_num
        self.search_total_num = args.search_total_num
        self.depth = args.dnn_depth
        self.units_list = [128] * self.depth
        self.learning_rate = args.learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.class_distinct = args.class_total_num
        self.batch_num = args.batch_num
        self.is_training = args.is_training
        self.init_graph()

    def init_graph(self):
        # 初始化喂入参数，placeholder名字要唯一，不能更改placeholder的任何信息
        self.video_ids_ph = tf.placeholder(tf.int32, shape=[None, None], name='video_ids')
        self.search_id_ph = tf.placeholder(tf.int32, shape=[None], name='search_id')
        self.age_ph = tf.placeholder(tf.float32, shape=[None], name='age')
        self.gender_ph = tf.placeholder(tf.float32, shape=[None], name='gender')
        self.label_ph = tf.placeholder(tf.float32, shape=[None], name='label_ph')

        # 初始化视频embedding、搜索条件的embedding，concat两个embedding和age、gender
        video_embedding = tf.get_variable('video_embedding', shape=[self.video_total_num], dtype=tf.float32,
                                          initializer=tf.variance_scaling_initializer())
        video_vecs = tf.nn.embedding_lookup(video_embedding, self.video_ids_ph)
        search_embedding = tf.get_variable(name='search_embedding', shape=[self.search_total_num], dtype=tf.float32,
                                           initializer=tf.variance_scaling_initializer())
        search_vec = tf.nn.embedding_lookup(search_embedding, self.search_id_ph)
        input = tf.concat([tf.reshape(tf.reduce_mean(video_vecs, axis=1), shape=[-1, 1]),
                           tf.reshape(search_vec, shape=[-1, 1]), tf.reshape(self.age_ph, shape=[-1, 1]),
                           tf.reshape(self.gender_ph, shape=[-1, 1])], axis=1)

        # 经过多层深度训练，层数根据mAP确定
        for i in range(self.depth):
            input = tf.layers.dense(inputs=input, units=self.units_list[i],
                                    kernel_regularizer=layers.l2_regularizer(0.001), activation=tf.nn.relu,
                                    name='fc{}'.format(i), trainable=self.is_training)
            input = tf.layers.batch_normalization(input, training=self.is_training, name='fc{}_bn'.format(i))
        output = input
        # 初始化类别（就是每个视频的标签，对应论文中的百万级）的embedding对应的：weights和bias
        weights = tf.get_variable('soft_weight', shape=[self.class_distinct, 128],
                                  initializer=tf.variance_scaling_initializer())
        biases = tf.get_variable('soft_bias', shape=[self.class_distinct],
                                 initializer=tf.variance_scaling_initializer())
        if not self.is_training:
            # 计算预测值
            self.logits_out = tf.matmul(output, tf.transpose(weights))
        else:
            # label必须二维的，但是biases却是一维的
            self.labels = tf.reshape(self.label_ph, shape=[-1, 1])
            # 计算损失, num_true=1代表负采样有一个正例，one-hot值为1。
            self.logits_out, self.labels_out = nn_impl._compute_sampled_logits(weights=weights, biases=biases,
                                                                               labels=self.labels,
                                                                               inputs=input, num_sampled=100,
                                                                               num_classes=self.class_distinct,
                                                                               num_true=1,
                                                                               sampled_values=None,
                                                                               remove_accidental_hits=True,
                                                                               partition_strategy="div",
                                                                               name="sampled_softmax_loss",
                                                                               seed=None)
            labels = array_ops.stop_gradient(self.labels_out, name="labels_stop_gradient")
            sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits_out)
            self.loss = tf.reduce_mean(sampled_losses)
            # 获得梯度下降优化器
            gradient_descent_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_var = tf.trainable_variables()
            clip_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), 5)
            self.gradient_descent = gradient_descent_optimizer.apply_gradients(zip(clip_gradients, train_var),
                                                                               global_step=self.global_step)

    def train(self, session, video_ids, search_id, age, gender, label):
        loss, _, step, logits, labels = session.run(
            [self.loss, self.gradient_descent, self.global_step, self.labels_out, self.logits_out],
            feed_dict={self.video_ids_ph: video_ids,
                       self.search_id_ph: search_id,
                       self.age_ph: age,
                       self.gender_ph: gender,
                       self.label_ph: label
                       })
        return loss, step, logits, labels

    def predict(self, session, video_ids, search_id, age, gender):
        result = session.run([self.logits_out],
                             feed_dict={self.video_ids_ph: video_ids,
                                        self.search_id_ph: search_id,
                                        self.age_ph: age,
                                        self.gender_ph: gender})
        return result

    def cal_acc(self, session, label, logit):
        labels = tf.one_hot(label, self.class_distinct, axis=1)
        correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(labels, 1))
        accuracy, labels, logits = session.run([tf.reduce_mean(tf.cast(correct_pred, tf.float32)), labels, logit])
        return accuracy, labels, logits

    def save(self, session, path):
        tf.train.Saver().save(session, path)

    def restore(self, session, path):
        tf.train.Saver().restore(session, path)


# 运行图
def run_graph(training=True):
    args = Args(is_training=training)
    # 可以增加gpu使用的设置
    with tf.Session() as session:
        recall_model = RecallModel(args)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        if args.is_training:
            for i in range(args.epoch):
                j = 0
                while True:
                    has_more, video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = get_batch(
                        page_size=args.batch_num)
                    if not has_more:
                        break
                    video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = list2array(
                        video_ids_batch, dtype='int32'), list2array(search_ids_batch, 'int32'), list2array(age_batch), list2array(
                        gender_batch), list2array(label_batch)
                    loss, step, logits, labels = recall_model.train(session, video_ids_batch, search_ids_batch,
                                                                    age_batch,
                                                                    gender_batch, label_batch)
                    print(loss, step)
                    if j % 1 == 0:
                        recall_model.save(session, args.checkpoint_dir + 'utube')
                    j += 1
        else:
            recall_model.restore(session, args.checkpoint_dir)
            has_more, video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = get_batch(2,
                page_size=args.batch_num)
            if has_more:
                video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = list2array(video_ids_batch, 'int32'), list2array(
                    search_ids_batch, 'int32'), list2array(age_batch), list2array(gender_batch), list2array(label_batch)
                logits = recall_model.predict(session, video_ids_batch, search_ids_batch, age_batch, gender_batch)[0]
                logits = tf.convert_to_tensor(logits, tf.float32, name='predict_logits')
                accuracy, labels, logits = recall_model.cal_acc(session, label=label_batch, logit=logits)
                print(accuracy)


if __name__ == '__main__':
    run_graph(training=False)
