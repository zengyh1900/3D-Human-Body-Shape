#!/usr/bin/python
# coding=utf-8

import numpy as np
import json
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":
    paras_file = "../bodyshape/parameter.json"
    cf_file = "../../data/miner/cfTable_02.npy"
    with open(paras_file, 'r') as f:
        paras = json.load(f)
    cfTable = np.load(open(cf_file, "rb"))
    m_str = paras["m_str"]

    m_str = ['体重', '身高', '颈围', '胸围', '下腰围', '髋围', '袖长',
             '腿长', '肩宽', '上身长', '上腰围', '臀围', '腰升', '臂长',
             '上臂围', '腕围', '裤长', '膝盖围度', '大腿腿围']

    n = 18
    x = np.arange(n)
    y = np.array(cfTable[1:, 0])
    print(len(x), len(y), len(m_str))

    # plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.bar(x, y, facecolor='blue', edgecolor='white')

    # for i in range(0, n):
    #     plt.text(x[i] + 0.4, y[i] + 0.05, m_str[i + 1],
    #              ha='center', va='bottom')

    plt.xticks(x, m_str[1:])
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.1))

    # savefig('../figures/bar_ex.png', dpi=48)
    plt.show()
