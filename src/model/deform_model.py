#!/usr/bin/python
# coding=utf-8

import sys
sys.path.append("..")
from dataProcess.dataModel import *
import numpy as np
import time


class deformModel:
    '''
        a measureModel show the PCA space of measure
        mapping measure_basis to vertex_basis to reconstruct body shape
    '''

    def __init__(self, data):
        self.TYPE = "deform-model"
        self.data = data
        self.d_basis_num = self.data.paras["d_basis_num"]
        self.demo_num = self.d_basis_num
        self.deformation = None

    # -------------------------------------------------------
    '''show deformation-based synthesize(PCA)'''
    # -------------------------------------------------------

    def show_d_pca(self):
        print(" [**] begin show deformation's PCA ...")
        start = time.time()
        for id in range(0, self.data.d_basis_num):
            for sign in [-1, +1]:
                alpha = np.zeros((self.data.d_basis_num, 1))
                alpha[id] = 3 * sign
                [vertex, n, f] = self.mapping(alpha)
                filename = self.data.ansPath + \
                    ('PC%d_%dsigma.obj' % (id, 3 * sign))
                self.data.save_obj(filename, vertex, self.data.o_faces)
        print(' [**] finish calculating coeff of data in %fs' %
              (time.time() - start))

    # -----------------------------------------------------------------------------------
    '''given coeff of pca deform_basis, return body shape'''
    # -----------------------------------------------------------------------------------

    def mapping(self, weight):
        weight = np.array(weight[:self.demo_num, :]).reshape(self.demo_num, 1)
        weight = self.data.d_pca_mean[
            :self.demo_num, :] + weight * self.data.d_pca_std[:self.demo_num, :]
        basis = self.data.d_basis[:, :self.data.d_basis_num]
        deformation = self.data.mean_deform + \
            np.matmul(basis, weight) * self.data.std_deform
        [v, n, f] = self.data.d_synthesize(deformation)
        return [v, n, f]

#############################################
'''test'''
#############################################
if __name__ == "__main__":
    filename = "../parameter.json"
    data = rawData(filename)
    bd = basisData(data)
    mark = Masker(data)

    # -----------------------------------
    model = dataModel(bd, mark)
    dm = deformModel(model)
    dm.show_d_pca()
