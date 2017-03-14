#!/usr/bin/python
# coding=utf-8

import numpy as np
import scipy.sparse.linalg
import scipy.sparse
import scipy
import time


# a measureModel show the PCA space of measure
# mapping measure_basis to vertex_basis to reconstruct body shape
class measureModel:

    def __init__(self, data, miner):
        self.TYPE = "measure-model"
        self.data = data
        self.miner = miner
        self.m_basis_num = self.data.paras["m_basis_num"]
        self.demo_num = self.m_basis_num
        self.M = self.M2V()
        self.deformation = None

    # calculate the mapping matrix from measures to vertex-based
    def M2V(self):
        print(' [**] begin load M2V matrix ... ')
        start = time.time()
        if self.data.paras["reload_M_mapping"]:
            W = numpy.array(self.data.v_coeff[:self.data.v_basis_num, ::]).reshape(
                self.data.v_basis_num, self.data.body_count)
            W = W.transpose().reshape((W.size, 1))
            M = np.array(self.pca.m_coeff[:self.m_basis_num, ::]).reshape(
                self.m_basis_num, self.data.body_count)
            M = self.data.build_equation(M, self.data.v_basis_num)
            # solve transform matrix
            MtM = M.transpose().dot(M)
            MtW = M.transpose().dot(W)
            ANS = np.array(scipy.sparse.linalg.spsolve(MtM, MtW))
            ANS.shape = ((self.data.v_basis_num, self.m_basis_num))
            np.save(open(self.data.NPYpath + 'M.npy', 'wb'), ANS)
            print(' [**] finish load_global_matrix between measures & vertex-based in %fs' %
                  (time.time() - start))
            return ANS
        else:
            print(' [**] finish load_global_matrix between measures & vertex-based in %fs' %
                  (time.time() - start))
            return np.load(open(self.data.NPYpath + 'M.npy', 'rb'))

    # show all pca of vertex-based space
    def show_m_pca(self):
        print(" [**] begin show vertex's PCA ...")
        start = time.time()
        for id in range(0, self.m_basis_num):
            for sign in [-1, +1]:
                alpha = np.zeros((self.m_basis_num, 1))
                alpha[id] = 3 * sign
                [v, n, f] = self.mapping(alpha)
                filename = self.data.ansPath + \
                    ('PC%d_%dsigma.obj' % (id, 3 * sign))
                self.data.save_obj(filename, v, self.data.o_faces)
        print(' [**] finish calculating coeff of data in %fs' %
              (time.time() - start))

    # mapping coeff of PCA measure_basis to PCA vertex_basis
    def mapping(self, measure):
        measure = np.array(measure[:self.demo_num, :])
        measure.shape = (self.demo_num, 1)
        measure *= np.array(self.pca.m_pca_std[:self.demo_num, :])
        measure += self.pca.m_pca_mean[:self.demo_num, :]
        weight = self.M.dot(measure)
        [v, n, f] = self.data.v_synthesize(weight)
        return [v, n, f]
