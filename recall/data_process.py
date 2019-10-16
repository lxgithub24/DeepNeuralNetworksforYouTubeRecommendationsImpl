# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:13
# @Author  : RIO
# @desc: TODO:DESC
import random
from utils.config import SuperPrams


# 获取每批数据
def get_batch(page_no=0, page_size=SuperPrams.batch_num):
    with open('../data/racall/feature.txt', 'r') as f:
        lines = f.readlines()
        if len(lines) <= page_no:
            return False, [], [], [], [], []
        video_ids_batch = []
        search_id_batch = []
        age_batch = []
        gender_batch = []
        label_batch = []
        for i in range(page_no * page_size, min(len(lines), page_size * page_no + page_size)):
            content = lines[i]
            info_list = content.split(',')
            video_ids = [int(i) if i else 0 for i in info_list[1].split('#')]
            video_ids_batch.append(enhance_data(video_ids))
            search_id = info_list[2].split('#')[0]
            search_id_batch.append(int(search_id) if search_id else 0)
            age_batch.append(int(info_list[3]))
            gender_batch.append(int(info_list[4]))
            label_batch.append(int(info_list[5].strip()))
        return True, video_ids_batch, search_id_batch, age_batch, gender_batch, label_batch


def enhance_data(data_list):
    """
    如果用户阅读的视频数不足100，会出现数据不整齐的情况。需要将数据补充整齐，做法如下：
    当前长度len，对于剩余（100-len）视频：其中的len*（100-len）/100从历史观看记录寻找；其余的从热门添加
    :param data_list: 待补充的视频列表
    :return: 增强后的视频列表
    """
    enhance_history_video_num = int(len(data_list)*(100-len(data_list))/100)
    return data_list + random.sample(data_list, enhance_history_video_num) + random.sample(range(SuperPrams.hot_video_max_num + 1), 100-len(data_list)-enhance_history_video_num)


# 获取总条数
def get_total_num():
    with open('../data/racall/feature.txt', 'r') as f:
        return len(f.readlines())


def get_hot_video(video_num=1):
    """
    # 获取热门视频，video：60%出自0-700热门视频
    :param video_num:获取的视频数
    :return: video_num个热门视频
    """
    # 获取视频数目小于热门最大编码
    if video_num < SuperPrams.hot_video_max_num + 1:
        return random.sample(range(SuperPrams.hot_video_max_num + 1), video_num)
    # 获取视频数目不小于热门最大编码
    return random.sample(range(SuperPrams.hot_video_max_num + 1), int(video_num * .6)) + random.sample(
        range(SuperPrams.hot_video_max_num, SuperPrams.video_total_num + 1), video_num - int(video_num * .6))
