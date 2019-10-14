# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:13
# @Author  : RIO
# @desc: TODO:DESC


# 获取每批数据
def get_batch(page_no=0, page_size=10000):
    with open('../data/feature.txt', 'r') as f:
        lines = f.readlines()
        if len(lines) <= page_no:
            return False, [], [], [], [], []
        video_ids_batch = []
        search_id_batch = []
        age_batch = []
        gender_batch = []
        label_batch = []
        for i in range(page_no * page_size, len(lines)):
            content = lines[i]
            info_list = content.split(',')
            video_ids_batch.append(info_list[1].split('#'))
            search_id_batch.append(info_list[2])
            age_batch.append(info_list[3])
            gender_batch.append(info_list[4])
            label_batch.append(info_list[5].strip())
        return True, video_ids_batch, search_id_batch, age_batch, gender_batch, label_batch


# 获取总条数
def get_total_num():
    with open('../data/feature.txt', 'r') as f:
        return len(f.readlines())
