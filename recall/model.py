# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 召回部分
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


class RecallModel:
    def __init__(self, video_total_num, video_embedding_length, search_total_num, search_embedding_length,
                 dnn_depth, num_class, class_embedding_length, learning_rate):
        self.video_total_num = video_total_num
        self.video_embedding_length = video_embedding_length
        self.search_total_num = search_total_num
        self.search_embedding_length = search_embedding_length
        self.depth = dnn_depth
        self.num_class = num_class
        self.class_embedding_length = class_embedding_length
        self.num_samples = 100  # 负采样样本数
        self.units_list = [128] * self.depth
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.init_graph()

    def init_graph(self):
        # 初始化喂入参数
        self.video_ids = tf.placeholder(tf.int32, shape=[None, None], name='video_ids')
        self.search_id = tf.placeholder(tf.int32, shape=[None, None], name='search_id')
        self.age = tf.placeholder(tf.int32, shape=[None], name='age')
        self.gender = tf.placeholder(tf.int32, shape=[None], name='gender')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')

        # 初始化视频embedding、搜索条件的embedding，concat两个embedding和age、gender
        video_embedding = tf.get_variable('video_embedding', shape=[self.video_total_num, self.video_embedding_length],
                                          dtype=tf.float32, initializer=tf.variance_scaling_initializer())
        video_vecs = tf.nn.embedding_lookup(video_embedding, self.video_ids)
        search_embedding = tf.get_variable(name='search_embedding',
                                           shape=[self.search_total_num, self.search_embedding_length])
        search_vec = tf.nn.embedding_lookup(search_embedding, self.search_id)
        input = tf.concat([tf.reduce_mean(video_vecs, axis=0), search_vec, self.age, self.gender], axis=1)

        # 经过多层深度训练，层数根据mAP确定
        for i in range(self.depth):
            input = tf.layers.dense(inputs=input, units=self.units_list[i],
                                    kernel_regularizer=layers.l2_regularizer(0.001), activation=tf.nn.relu,
                                    name='fc{}'.format(i))
            input = tf.layers.batch_normalization(input, training=True, name='fc{}_bn'.format(i))

        # 初始化类别（就是每个视频的标签，对应论文中的百万级）的embedding：weights(这个名字起的很容易误解为权重)。bias
        weights = tf.get_variable('soft_weight', shape=[self.num_class, self.class_embedding_length],
                                  initializer=tf.variance_scaling_initializer())
        biases = tf.get_variable('soft_bias', shape=[self.num_class], initializer=tf.variance_scaling_initializer())
        # 计算损失, num_true=1代表负采样有一个正例，one-hot值为1。
        self.logits, labels = nn_impl._compute_sampled_logits(weights=weights, biases=biases, labels=self.label,
                                                              inputs=input, num_sampled=100,
                                                              num_classes=set(self.label).__sizeof__(), num_true=1,
                                                              sampled_values=None, remove_accidental_hits=True,
                                                              partition_strategy="mod", name="sampled_softmax_loss",
                                                              seed=None)
        labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
        sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits)
        self.loss = tf.reduce_mean(sampled_losses)
        # 获得梯度下降优化器
        gradient_descent_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        train_var = tf.trainable_variables()
        clip_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), 5)
        self.gradient_descent = gradient_descent_optimizer.apply_gradients(zip(clip_gradients, train_var),
                                                                           global_step=self.global_step)

    def train(self, session, video_ids, search_id, age, gender, label):
        loss, _, step = session.run([self.loss, self.gradient_descent, self.global_step],
                                    feed_dict={self.video_ids: video_ids, self.search_id: search_id, self.age: age, self.gender: gender, self.label:label})
        return loss, step

    def predict(self, session, video_ids, search_id, age, gender):
        result = session.run([self.logits],
                             feed_dict={video_ids: video_ids, search_id: search_id, age: age, gender: gender})
        return result

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
        recall_model = RecallModel(video_total_num, video_embedding_length, search_total_num, search_embedding_length,
                                   dnn_depth, num_class, class_embedding_length, learning_rate)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        count = int(len(data['label']) / batch_num)
        for i in range(epoch):
            for j in range(count):
                video_ids, search_id, age, gender = get_batch()
                loss, step = recall_model.train(session, video_ids, search_id, age, gender, label)
                if j % 100 == 0:
                    recall_model.save(session, checkpoint_dir)


if __name__ == '__main__':
    run_graph()
