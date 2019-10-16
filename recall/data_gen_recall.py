# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 10:50
# @Author  : RIO
# @desc: TODO:DESC
import random
from collections import Counter
from utils.config import SuperPrams

uid_max = SuperPrams.user_total_num - 1
video_id_max = SuperPrams.video_total_num - 1
# search_id_max = 100000
search_id_max = SuperPrams.search_total_num - 1
class_id_max = SuperPrams.class_total_num - 1
# feature_id_max = 80000
feature_id_max = SuperPrams.feature_id_total_num - 1


# uid
def get_uid():
    random_choice_list = [1] * 14 + [2] * 5 + [3]
    uid_period = random.choice(random_choice_list)
    # 0-199:70%;
    if uid_period == 1:
        return random.randint(0, 199)
    # 200-500:25%;
    elif uid_period == 2:
        return random.randint(200, 499)
    # 500-:5%
    return random.randint(500, uid_max)


# video_id list
def get_video_id():
    # video：60%出自0-700热门视频
    def gen_video():
        _random_choice_list = [1] * 3 + [2] * 2
        video_period = random.choice(_random_choice_list)
        if video_period == 1:
            return random.randint(0, 700)
        return random.randint(701, video_id_max)

    # video数：45-50：80%；5-45：18%；其他2%
    def get_video_len():
        random_choice_list = [1] * 40 + [2] * 9 + [3]
        video_len_period = random.choice(random_choice_list)
        if video_len_period == 1:
            video_len = random.randint(45, 50)
        elif video_len_period == 2:
            video_len = random.randint(5, 45)
        else:
            video_len = random.randint(0, 100)
        return video_len

    video_list = []
    video_len = get_video_len()
    for _ in range(video_len):
        video_list.append(str(gen_video()))
    return list(set(video_list))


# search_id
def get_search_id():
    # 前40占搜索的40%，40-100占搜索的40%, 其余20%
    def gen_search_id():
        random_choice_list = [1] * 2 + [2] * 2 + [3] * 1
        search_id_period = random.choice(random_choice_list)
        if search_id_period == 1:
            search_id = random.randint(0, 40)
        elif search_id_period == 2:
            search_id = random.randint(41, 100)
        else:
            search_id = random.randint(100, search_id_max)
        return search_id

    # search数：0-10：80%；10-50：20%
    def gen_search_len():
        random_choice_list = [1] * 4 + [2] * 1
        search_len_period = random.choice(random_choice_list)
        if search_len_period == 1:
            search_len = random.randint(0, 10)
        else:
            search_len = random.randint(11, 50)
        return search_len

    search_list = []
    video_len = gen_search_len()
    for _ in range(video_len):
        search_list.append(str(gen_search_id()))
    return list(set(search_list))


def get_age():
    # 年龄：40岁以上：15%；18-40：75%；0-18：10%
    random_choice_list = [1] * 3 + [2] * 15 + [3] * 2
    age_period = random.choice(random_choice_list)
    if age_period == 1:
        return random.randint(40, 100)
    elif age_period == 2:
        return random.randint(18, 40)
    return random.randint(0, 18)


def get_gender():
    # 性别：男（1）：女（2）=9：1
    random_choice_list = [1] * 9 + [2]
    return random.choice(random_choice_list)


def get_label(video_id):
    # video：0-700热门视频对应的class分布为：0-100占40%；101-500占40%；501-class_id_max占20%
    # video：700-9999热门视频对应的class分布为：0-100占30%；101-500占30%；501-class_id_max占40%
    def get_class(first, second, third):
        random_choice_list = [1] * first + [2] * second + [3] * third
        class_period = random.choice(random_choice_list)
        if class_period == 1:
            return random.randint(0, 100)
        elif class_period == 2:
            return random.randint(101, 500)
        else:
            return random.randint(501, class_id_max)

    if video_id > 700:
        class_id = get_class(2, 2, 1)
    else:
        class_id = get_class(3, 3, 4)
    return class_id


# 生成最终的特征文件
def gen_feature(has_label=True):
    def get_sample(i, has_label=True):
        print(i)
        uid = get_uid()
        video_ids = get_video_id()
        search_ids = get_search_id()
        age = get_age()
        gender = get_gender()
        if not has_label:
            label_list = [str(get_label(i)) for i in range(video_id_max + 1)]
            with open('../data/racall/videoid_videoclass.txt', 'w') as f:
                f.writelines(','.join(label_list))
        else:
            with open('../data/racall/videoid_videoclass.txt', 'r') as f:
                label_list = f.readlines()[0].strip().split(',')
        if len(video_ids):
            count_list = [label_list[int(i)] for i in video_ids]
            y_label = Counter(count_list).most_common(1)[0][0]
        else:
            y_label = random.choice(label_list)
        return str(uid) + ',' + '#'.join(video_ids) + ',' + '#'.join(search_ids) + ',' + str(age) + ',' + str(
            gender) + ',' + str(y_label)

    with open('../data/racall/feature.txt', 'w') as f:
        f.writelines('\n'.join([get_sample(i, has_label) for i in range(feature_id_max+1)]))


if __name__ == '__main__':
    gen_feature(False)
