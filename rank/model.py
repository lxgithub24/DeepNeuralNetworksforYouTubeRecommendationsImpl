# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 排序部分
import tensorflow as tf
from tensorflow.contrib import layers


class RankModel:
    def __init__(self, video_total_num, video_embedding_length, search_total_num, search_embedding_length, dnn_depth,
                 num_class, class_embedding_length, label,
                 learning_rate):
        # 总视频数
        self.video_total_num = video_total_num
        # 视频的embedding长度
        self.video_embedding_length = video_embedding_length
        # 总搜索数
        self.search_total_num = search_total_num
        # 搜索条件的embedding长度
        self.search_embedding_length = search_embedding_length
        # 网络深度
        self.depth = dnn_depth
        # 视频的类别总数
        self.num_class = num_class
        # 类别的embedding长度
        self.class_embedding_length = class_embedding_length
        # label
        self.label = label
        # 负采样样本数
        self.num_neg_samples = 100
        # 每层神经元数
        self.units_list = [32] * self.depth
        self.learning_rate = learning_rate
        # 梯度下降记录
        self.global_step = tf.Variable(0, trainable=False)
        self.init_graph()

    def init_graph(self):
        # 初始化喂入参数
        # 最后观看的6个视频
        self.last_watch_video_ids = tf.placeholder(tf.int32, shape=[None, None], name='last_watch_video_ids')
        # 最后一次搜索的embedding，空则随机值
        self.last_search = tf.placeholder(tf.float32, shape=[None], name='last_search')
        # 距离上次观看的时间
        self.time_since_last_watch = tf.placeholder(tf.float32, shape=[None], name='time_since_last_watch')
        # weighted logistic regression每条训练数据y所做的权重。label=0样本的值为1；label=1值为视频观看进度百分比*100
        self.lr_weights = tf.placeholder(tf.int32, shape=[None], name='lr_weights')

        # 初始化视频embedding、搜索条件的embedding，concat两个embedding和age、gender
        video_embedding = tf.get_variable('video_embedding', shape=[self.video_total_num, self.video_embedding_length],
                                          dtype=tf.float32, initializer=tf.variance_scaling_initializer())
        last_watch_video_vec = tf.nn.embedding_lookup(video_embedding, self.last_watch_video_ids)

        input = tf.concat([tf.reduce_mean(last_watch_video_vec, axis=0), self.last_search, self.time_since_last_watch],
                          axis=1)

        # 经过多层深度训练，层数根据mAP确定
        for i in range(self.depth):
            input = tf.layers.dense(inputs=input, units=self.units_list[i],
                                    kernel_regularizer=layers.l2_regularizer(0.001), activation=tf.nn.relu,
                                    name='fc{}'.format(i))
            input = tf.layers.batch_normalization(input, training=True, name='fc{}_bn'.format(i))
        self.output = input
        # 初始化w和b
        self.weights = tf.Variable(tf.random_normal([self.output.shape[0], 1], 0.0))
        # y = 1/(1+exp(-w*x))
        self.output = tf.nn.sigmoid(tf.multiply(self.weights, self.output))
        self.loss = self.label * tf.log(self.output) + (1 - self.label) * tf.log(1 - self.output) + tf.log(
            self.lr_weights)

        # 获得梯度下降优化器
        gradient_descent_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        train_var = tf.trainable_variables()
        clip_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), 5)
        self.gradient_descent = gradient_descent_optimizer.apply_gradients(zip(clip_gradients, train_var),
                                                                           global_step=self.global_step)

    def train(self, session, last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights):
        loss, _, step = session.run([self.loss, self.gradient_descent, self.global_step],
                                    feed_dict={self.last_watch_video_ids: last_watch_video_ids,
                                               self.last_search: last_search,
                                               self.time_since_last_watch: time_since_last_watch,
                                               self.label: label, self.lr_weights: lr_weights})
        return loss, step

    def predict(self, session, last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights):
        w, x = session.run([self.weights, self.output],
                           feed_dict={self.last_watch_video_ids: last_watch_video_ids,
                                      self.last_search: last_search,
                                      self.time_since_last_watch: time_since_last_watch})
        return tf.exp(tf.multiply(w, x))

    def save(self, session, path):
        tf.train.Saver().save(session, path)

    def restore(self, session, path):
        tf.train.Saver().restore(session, path)


# 获取每批数据
def get_batch():
    pass
    return 1


# 运行图
def run_graph():
    checkpoint_dir, epoch, batch_num, video_total_num, video_embedding_length, search_total_num, search_embedding_length, dnn_depth, num_class, class_embedding_length, label, learning_rate = []
    data = {}
    # 可以增加gpu使用的设置
    with tf.Session() as session:
        rank_model = RankModel(video_total_num, video_embedding_length, search_total_num, search_embedding_length,
                                   dnn_depth, num_class, class_embedding_length, label, learning_rate)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        count = int(len(data['label']) / batch_num)
        for i in range(epoch):
            for j in range(count):
                last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights = get_batch()
                loss, step = rank_model.train(session, last_watch_video_ids, last_search, time_since_last_watch, label, lr_weights)
                if j % 100 == 0:
                    rank_model.save(session, checkpoint_dir)


if __name__ == '__main__':
    run_graph()
