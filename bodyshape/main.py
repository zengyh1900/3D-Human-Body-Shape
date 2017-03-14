#!/usr/bin/python
# coding=utf-8

from myutils.meta import *
from myutils.miner import *
from myutils.reshaper import *
from model.vertex_model import *


# test for myutils.meta
def test_meta(paras):
    male = MetaData(paras, 1)
    female = MetaData(paras, 2)
    return [male, female]


# test for myutils.miner
def test_miner(paras):
    male = MetaData(paras, 1)
    male_miner = Miner(male)
    male_miner.test()

    female = MetaData(paras, 2)
    female_miner = Miner(female)
    female_miner.test()


# test for myutils.reshaper
def test_reshaper(paras):
    test_vertex_model(paras)


# test for vertex_model
def test_vertex_model(paras):
    male_data = MetaData(paras, 1)
    male = Reshaper(male_data)
    # female_data = MetaData(paras, 2)
    # female = Reshaper(female_data)
    # model = VertexModel(male, female)
    # model.show_v_pca()


if __name__ == "__main__":
    paras = "parameter.json"
    test_vertex_model(paras)

    # a = numpy.array([i for i in range(0, 24)]).reshape(2, 3, 4)
    # print(a)
    # a.shape = (2, 12)
    # a = a.transpose()
    # print(a)
    # a = a.transpose()
    # a.shape = (2, 3, 4)
    # print(a)

