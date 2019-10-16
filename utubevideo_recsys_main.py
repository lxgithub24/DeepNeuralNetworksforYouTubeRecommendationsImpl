# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 工程入口
from recall.model import run_graph as recall_model
from rank.model import run_graph as rank_model

# 召回
recall_model()
# 排序
rank_model()