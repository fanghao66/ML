# -*- coding: utf-8 -*-
import time

import numpy as np

import faiss
from sklearn.neighbors import KDTree

np.random.seed(1234)  # make reproducible 随机数种子给定

"""
NOTE:
1. 用来构建faiss索引的数据类型必须是float32类型
2. 其它向量检索库：
    http://www.javashuo.com/article/p-apcgrtch-nu.html
"""

# 产生一个随机数据
d = 64  # dimension 给定向量维度大小

nb = 100000  # database size
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

nq = 10000  # nb of queries
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

k = 4  # we want to see 4 nearest neighbors
search_xb = xb[:5]  # [5,64] 待检索的向量

#KD-Tree
def t0():
    print("=" * 100)
    _t0 = time.time()
    # 构建索引
    index = KDTree(xb)  # 索引对象创建 + 索引构建
    _t1 = time.time()

    # 检索数据
    k = 4  # we want to see 4 nearest neighbors
    D, I = index.query(search_xb, k, return_distance=True)  # 针对每个待检索的向量(每个行向量)返回K个最相似的向量
    print(I)
    print(D)
    D, I = index.query(xq, k, return_distance=True)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    _t2 = time.time()
    print(f"耗时:{_t2 - _t1} -- {_t1 - _t0}")


# IndexFlatL2
def t1():
    print("=" * 100)
    _t0 = time.time()
    # 构建索引
    index = faiss.IndexFlatL2(d)  # build the index 创建一个索引对象
    print(index.is_trained)  # 用来判断是否需要进行训练 --> 索引是否有一些特殊的参数需要从数据中获取
    index.add(xb)  # add vectors to the index 构建索引
    print(index.ntotal)
    _t1 = time.time()

    # 检索数据
    D, I = index.search(search_xb, k)  # 针对每个待检索的向量(每个行向量)返回K个最相似的向量
    print(I)
    print(D)
    D, I = index.search(xq, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    _t2 = time.time()
    print(f"耗时:{_t2 - _t1} -- {_t1 - _t0}")


# IndexIVFFlat
def t2():
    print("=" * 100)
    _t0 = time.time()
    nlist = 100  # 需要学习的中心点数目
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)  # 倒排索引：类似聚类
    print(index.is_trained)
    assert not index.is_trained
    index.train(xb)  # 当前索引需要进行参数的训练/学习
    assert index.is_trained
    index.add(xb)  # 在训练后添加索引，不会改变训练结果，但是数据分布发生了变化
    _t1 = time.time()

    D, I = index.search(search_xb, k)  # 针对每个待检索的向量(每个行向量)返回K个最相似的向量
    print(I)
    print(D)
    D, I = index.search(xq, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    _t2 = time.time()
    print(f"耗时:{_t2 - _t1} -- {_t1 - _t0}")

    print("-" * 50)
    x0 = np.random.random((1, d)).astype('float32')
    D, I = index.search(x0, k)  # actual search
    print(I)
    # 将x0添加到索引中, 索引中的数据就增加一个
    index.add(x0)  # 当一个索引是需要train的时候，建议不要add新的数据(未参与train过程的数据)
    # 重新检索
    D, I = index.search(x0, k)  # actual search
    print(I)
    D, I = index.search(search_xb, k)  # 针对每个待检索的向量(每个行向量)返回K个最相似的向量
    print(I)
    print(D)

#HNSW128
def t3():
    print("=" * 100)
    _t0 = time.time()
    dim, measure = d, faiss.METRIC_L2  # 给定维度和相似度度量方式
    param = 'HNSW128'
    index = faiss.index_factory(dim, param, measure)  # 创建索引对象
    print(index)
    print(index.is_trained)
    if not index.is_trained:
        print("表示当前索引需要进行training学习")
        index.train(xb)
    index.add(xb)  # 添加数据构建索引
    _t1 = time.time()

    D, I = index.search(search_xb, k)  # 针对每个待检索的向量(每个行向量)返回K个最相似的向量
    print(I)
    print(D)
    D, I = index.search(xq, k)  # 检索目标向量的最近邻，返回k个
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    _t2 = time.time()
    print(f"耗时:{_t2 - _t1} -- {_t1 - _t0}")

    print("-" * 50)
    x0 = np.random.random((1, d)).astype('float32')
    D, I = index.search(x0, k)  # actual search
    print(I)
    # 将x0添加到索引中, 索引中的数据就增加一个
    index.add(x0)
    # 重新检索
    D, I = index.search(x0, k)  # actual search
    print(I)


if __name__ == '__main__':
    # t0()
    # t1()
    # t2()
    t3()
