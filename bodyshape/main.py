#!/usr/bin/python
# coding=utf-8

from myutils.meta import *
from myutils.miner import *
from myutils.reshaper import *
from model.vertex_model import *
from model.measure_model import *
from model.deform_model import *
from model.vertex_global import *


paras = "parameter.json"
male = MetaData(paras, 1)
female = MetaData(paras, 2)

male_miner = Miner(male)
female_miner = Miner(female)


male_body = Reshaper(male)
female_body = Reshaper(female)


# test for vertex_model
def test_vertex_model():
    model = VertexModel(male_body, female_body)
    model.show_v_pca()


# test for measure_model
def test_measure_model():
    model = MeasureModel(male_body, female_body)
    model.show_m_pca()


# test for deform_Model
def test_deform_model():
    model = DeformModel(male_body, female_body)
    model.show_d_pca()


def test_vertex_global():
    model = VertexGlobal(male_body, female_body)
    model.v_rebuild()


if __name__ == "__main__":
    test_vertex_global()
