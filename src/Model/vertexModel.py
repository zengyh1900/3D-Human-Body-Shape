#!/usr/bin/python
#coding=utf-8

import sys
sys.path.append("..")
from dataProcess.dataModel import *
import numpy as np
import time

class vertexModel:
    '''
        a measureModel show the PCA space of measure
        mapping measure_basis to vertex_basis to reconstruct body shape
    '''
    def __init__(self, data):
        self.TYPE = "vertex-model"
        self.data = data
        self.v_basis_num = self.data.paras["v_basis_num"]
        self.demo_num = self.v_basis_num
        self.deformation = None

    # -------------------------------------------------------
    '''show all pca of vertex-based space '''
    # -------------------------------------------------------
    def show_v_pca(self):
        print (" [**] begin show vertex's PCA ...")
        start = time.time()
        for id in range(0, self.data.v_basis_num):
            for sign in [-1, +1]:
                alpha = np.zeros((self.data.v_basis_num, 1))
                alpha[id] = 3 * sign
                [v, n, f]  = self.data.v_synthesize(alpha)
                filename = self.data.ansPath + ('PC%d_%dsigma.obj' % (id, 3 * sign))
                self.data.save_obj(filename, v, f+1)
        print (' [**] finish calculating coeff of data in %fs' % (time.time() - start))

    # -----------------------------------------------------------------------------------
    '''given coeff of pca_vertex_basis, return body shape '''
    # -----------------------------------------------------------------------------------
    def mapping(self, coeff):
        coeff = np.array(coeff[:self.demo_num,:]).reshape(self.demo_num, 1)
        coeff = self.data.v_pca_mean[:self.demo_num,:] + coeff * self.data.v_pca_std[:self.demo_num,:]
        [v, n, f]  = self.data.v_synthesize(coeff)
        return [v,n,f]


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
    vm = vertexModel(model)
    vm.show_v_pca()