#!/usr/bin/python
# coding=utf-8

from myutils.meta import *
from myutils.miner import *
from myutils.reshaper import *
from model.vertex_model import *
from model.measure_model import *


paras = "parameter.json"
male = MetaData(paras, 1)
female = MetaData(paras, 2)


# test for myutils.meta
def test_meta():
    pass


# test for myutils.miner
def test_miner():
    male_miner = Miner(male)
    male_miner.test()

    female_miner = Miner(female)
    female_miner.test()


# test for myutils.reshaper
def test_reshaper():
    test_vertex_model(paras)


# test for vertex_model
def test_vertex_model():
    male_body = Reshaper(male)
    female_body = Reshaper(female)
    model = VertexModel(male_body, female_body)
    model.show_v_pca()


# test for measure_model
def test_measure_model():
    male_body = Reshaper(male)
    female_body = Reshaper(female)
    model = MeasureModel(male_body, female_body)
    model.show_v_pca()


if __name__ == "__main__":
    test_measure_model()

    # a = numpy.array([i for i in range(0, 24)]).reshape(2, 3, 4)
    # print(a)
    # a.shape = (2, 12)
    # a = a.transpose()
    # print(a)
    # a = a.transpose()
    # a.shape = (2, 3, 4)
    # print(a)
