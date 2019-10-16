# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 10:25
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np


def list2array(input, dtype='f'):
    if dtype == 'f':
        return np.array(input, dtype=np.float32)
    return np.array(input, dtype=np.int32)


if __name__ == '__main__':
    # has_more, video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = get_batch(
    #     page_size=500)
    # # video_ids_batch, search_ids_batch, age_batch, gender_batch, label_batch = list2array(video_ids_batch), list2array(
    # #     search_ids_batch), list2array(age_batch) \
    # #     , list2array(gender_batch), list2array(label_batch)
    # print(label_batch)
    # label_batch = list2array(label_batch)
    a = [[1,2,3],[1,2,3]]
    b = np.array(a,dtype=np.float32)
    print(b)
