# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 排序部分
import tensorflow as tf
from tensorflow.contrib import layers
from utils.config import SuperPrams
from rank.data_process import get_batch
from utils.utils import list2array


class Args(SuperPrams):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.dnn_depth = 3
        self.learning_rate = 0.01
        self.epoch = 10
        self.checkpoint_dir = '../data/rank/checkpoint_dir/'


class RankModel:
    def __init__(self, args):
        # 总视频数
        self.video_total_num = args.video_total_num
        # 总搜索数
        self.search_total_num = args.search_total_num
        # 网络深度
        self.depth = args.dnn_depth
        # 视频的类别总数
        self.class_distinct = args.class_total_num
        # 每层神经元数
        self.units_list = [32] * self.depth
        # 学习率
        self.learning_rate = args.learning_rate
        # 梯度下降记录
        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = args.is_training
        self.batch_num = args.batch_num
        self.init_graph()

    def init_graph(self):
        # 初始化喂入参数
        # 最后观看的6个视频
        self.last_watch_video_ids_ph = tf.placeholder(tf.int32, shape=[None, None], name='last_watch_video_ids')
        # 最后一次搜索的embedding，空则随机值
        self.last_search_ph = tf.placeholder(tf.float32, shape=[None], name='last_search')
        # 距离上次观看的时间
        self.time_since_last_watch_ph = tf.placeholder(tf.float32, shape=[None], name='time_since_last_watch')
        # weighted logistic regression每条训练数据y所做的权重。label=0样本的值为1；label=1值为视频观看进度百分比*100
        self.lr_weights_ph = tf.placeholder(tf.float32, shape=[None], name='lr_weights')
        self.label_ph = tf.placeholder(tf.float32, shape=[None], name='label_ph')

        # 初始化视频embedding、搜索条件的embedding，concat两个embedding和age、gender
        video_embedding = tf.get_variable('video_embedding', shape=[self.video_total_num],
                                          dtype=tf.float32, initializer=tf.variance_scaling_initializer())
        last_watch_video_vec = tf.nn.embedding_lookup(video_embedding, self.last_watch_video_ids_ph)

        input = tf.concat([tf.reshape(tf.reduce_mean(last_watch_video_vec, axis=1), shape=[-1, 1]),
                           tf.reshape(self.last_search_ph, shape=[-1, 1]),
                           tf.reshape(self.time_since_last_watch_ph, shape=[-1, 1])],
                          axis=1)

        # 经过多层深度训练，层数根据mAP确定
        for i in range(self.depth):
            input = tf.layers.dense(inputs=input, units=self.units_list[i],
                                    kernel_regularizer=layers.l2_regularizer(0.001), activation=tf.nn.relu,
                                    name='fc{}'.format(i), trainable=self.is_training)
            input = tf.layers.batch_normalization(input, training=self.is_training, name='fc{}_bn'.format(i))
        self.X = input
        # 初始化w和b
        self.weights = tf.get_variable('soft_weight', shape=[32, 1],
                                       initializer=tf.variance_scaling_initializer())
        # y = 1/(1+exp(-w*x))
        self.output = tf.nn.sigmoid(tf.matmul(self.X, self.weights))
        self.loss = - tf.reduce_mean(self.label_ph * tf.log(self.output + 1e-24) + (1 - self.label_ph) * tf.log(
            1 - self.output + 1e-24) + tf.log(self.lr_weights_ph + 1e-24))

        # 获得梯度下降优化器
        gradient_descent_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_var = tf.trainable_variables()
        clip_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), 5)
        self.gradient_descent = gradient_descent_optimizer.apply_gradients(zip(clip_gradients, train_var),
                                                                           global_step=self.global_step)

    def train(self, session, last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights):
        loss, _, step = session.run([self.loss, self.gradient_descent, self.global_step],
                                    feed_dict={self.last_watch_video_ids_ph: last_watch_video_ids,
                                               self.last_search_ph: last_search,
                                               self.time_since_last_watch_ph: time_since_last_watch,
                                               self.label_ph: label, self.lr_weights_ph: lr_weights})
        return loss, step

    def predict(self, session, last_watch_video_ids, last_search, time_since_last_watch):
        w, x = session.run([self.weights, self.X], feed_dict={self.last_watch_video_ids_ph: last_watch_video_ids,
                                                              self.last_search_ph: last_search,
                                                              self.time_since_last_watch_ph: time_since_last_watch})
        return tf.exp(tf.matmul(x, w))

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
    rank_model = RankModel(args)

    # 可以增加gpu使用的设置
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        if args.is_training:
            for i in range(args.epoch):
                j = 0
                while True:
                    has_more, last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights = get_batch(
                        page_size=args.batch_num)
                    if not has_more:
                        break
                    last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights = list2array(
                        last_watch_video_ids, dtype='int32'), list2array(last_search, 'int32'), list2array(
                        time_since_last_watch), list2array(
                        label), list2array(lr_weights)
                    loss, step = rank_model.train(session, last_watch_video_ids, last_search, time_since_last_watch,
                                                  label, lr_weights)
                    print(loss, step)
                    if j % 1 == 0:
                        rank_model.save(session, args.checkpoint_dir + 'utube')
                    j += 1
        else:
            rank_model.restore(session, args.checkpoint_dir)
            has_more, last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights = get_batch(
                page_size=args.batch_num)
            if has_more:
                last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights = list2array(
                    last_watch_video_ids, dtype='int32'), list2array(last_search, 'int32'), list2array(
                    time_since_last_watch), list2array(label), list2array(lr_weights)
                logits = rank_model.predict(session, last_watch_video_ids, last_search, time_since_last_watch)[0]
                accuracy, labels, logits = rank_model.cal_acc(session, label=label, logit=logits)
                print(accuracy)


if __name__ == '__main__':
    run_graph(training=True)
