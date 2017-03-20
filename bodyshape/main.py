#!/usr/bin/python
# coding=utf-8

from myutils import *
from model import *
from visual import *
from PyQt4 import QtGui
import sys


paras = "parameter.json"


# test for vertex_model
def show_pca():
    male = MetaData(paras, 1)
    male_body = Reshaper(male)

    female = MetaData(paras, 2)
    female_body = Reshaper(female)

    vm = VertexModel(male_body, female_body)
    vm.show_v_pca()
    dm = DeformModel(male_body, female_body)
    dm.show_d_pca()
    mm = MeasureModel(male_body, female_body)
    mm.show_m_pca()


def rebuild():
    male = MetaData(paras, 1)
    male_body = Reshaper(male)

    female = MetaData(paras, 2)
    female_body = Reshaper(female)
    vg = VertexGlobal(male_body, female_body)
    vg.v_rebuild()
    dg = DeformGlobal(male_body, female_body)
    dg.d_rebuild()
    dl = DeformLocal(male_body, female_body)
    dl.local_rebuild()


def show_app():
    app = QtGui.QApplication(sys.argv)

    win = HumanShapeAnalysisDemo(paras)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    rebuild()
