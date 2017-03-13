#!/usr/bin/python
# coding=utf-8

import scipy.sparse
import scipy.sparse.linalg
from meta import *
from basis import *
from mask import *


# A dataModel provide basic function and data
class DataModel:
    __metaclass__ = Singleton

    def __init__(self, basic, masker):
        self.data = basic.data
        self.basic = basic
        self.masker = masker
        self.loadData()
        switcher = {
            1: lambda: [0, 0],
            2: self.v_part_dad,
            3: self.all_dav,
            4: self.part_dav,
        }
        [self.m2v_A, self.m2v_lu] = switcher.get(
            self.paras['Hybrid_method'], lambda: [0, 0])()

    # --------------------------------------------------------------------------------
    '''load all data it need '''
    # -------------------------------------------------------------------------------

    def loadData(self):
        self.paras = self.data.paras
        self.NPYpath = self.paras["dataPath"] + "NPYdata/"
        self.ansPath = self.paras['ansPath']
        [self.o_normals, self.v_basis_num, self.d_basis_num] = \
            [self.data.o_normals, self.basic.v_basis_num, self.basic.d_basis_num]
        [self.body_count, self.measure_num, self.vertex_num, self.face_num] = \
            [self.data.body_count, self.data.measure_num,
                self.data.vertex_num, self.data.face_num]
        [self.o_file_list, self.o_faces, self.o_vertex, self.t_vertex, self.mean_vertex, self.std_vertex] = \
            [self.data.o_file_list, self.data.o_faces, self.data.o_vertex,
                self.data.t_vertex, self.data.mean_vertex, self.data.std_vertex]
        [self.o_deform, self.t_deform, self.mean_deform, self.std_deform] = \
            [self.basic.o_deform, self.basic.t_deform,
                self.basic.mean_deform, self.basic.std_deform]
        [self.v_basis, self.v_pca_mean, self.v_pca_std, self.v_coeff] = \
            [self.basic.v_basis, self.basic.v_pca_mean,
                self.basic.v_pca_std, self.basic.v_coeff]
        [self.d_inv_mean, self.d_basis, self.d_pca_mean, self.d_pca_std, self.d_coeff] = \
            [self.basic.d_inv_mean, self.basic.d_basis, self.basic.d_pca_mean,
                self.basic.d_pca_std, self.basic.d_coeff]
        [self.measure_str, self.o_measures, self.t_measures, self.mean_measures, self.std_measures] = \
            [self.data.measure_str, self.data.o_measures, self.data.t_measures,
                self.data.mean_measures, self.data.std_measures]
        [self.build_equation, self.construct_coeff_mat] = [
            self.data.build_equation, self.basic.construct_coeff_mat]
        [self.v_synthesize, self.d_synthesize, self.save_obj, self.calc_measures] = \
            [self.basic.v_synthesize, self.basic.d_synthesize,
                self.data.save_obj, self.data.calc_measures]
        [self.save_NPY, self.load_NPY] = [
            self.data.save_NPY, self.data.load_NPY]
        [self.color_list, self.color_set] = [
            self.masker.color_list, self.masker.color_set]
        [self.p2m, self.p2p, self.p2f, self.m2p, self.m2f, self.m2v, self.mask] = \
            [self.masker.p2m, self.masker.p2p, self.masker.p2f, self.masker.m2p,
                self.masker.m2f, self.masker.m2v, self.masker.mask]
        self.face_vertex = self.masker.face_vertex


# test for this module
if __name__ == "__main__":
    filename = "../parameter.json"
    data = rawData(filename)
    data = basisData(data)
    masker = Masker(data)
    model = dataModel(data)
