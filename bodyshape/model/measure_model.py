#!/usr/bin/python
# coding=utf-8

import numpy
import scipy.sparse.linalg
import scipy.sparse
import scipy
import time


# a measureModel show the PCA space of measure
# mapping measure_basis to vertex_basis to reconstruct body shape
class MeasureModel:

    def __init__(self, male, female):
        self.TYPE = "measure-model"
        self.body = [male, female]
        self.M = self.M2V()
        self.current_body = self.body[0]

        self.m_basis_num = self.current_body.paras["m_basis_num"]
        self.ans_path = self.current_body.ans_path + "measure_model/"
        self.demo_num = self.m_basis_num
        self.deformation = None

    # set current body
    def set_body(self, flag):
        self.current_body = self.body[flag - 1]

    # calculate the mapping matrix from measures to vertex-based
    def M2V(self):
        print(' [**] begin load M2V matrix ... ')
        start = time.time()
        M = []
        if self.current_body.paras["reload_M_mapping"]:
            for body in self.body:
                W = self.
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

        print("  finish load M2V matrix in %fs" % (time.time() - start))
        return M

    # show all pca of vertex-based space
    def show_m_pca(self):
        print(" [**] begin show measure's PCA ...")
        start = time.time()
        names = [self.ans_path + "01/", self.ans_path + "02/"]
        for i in range(0, 2):
            self.set_body(i + 1)
            for j in range(0, self.m_basis_num):
                for sign in [-1, +1]:
                    alpha = numpy.zeros((self.m_basis_num, 1))
                    alpha[j] = 3 * sign
                    [v, n, f] = self.mapping(alpha)
                    fname = names[i] + ('PC%d_%dsigma.obj' % (j, 3 * sign))
                    self.current_body.save_obj(fname, v, f + 1)
        print(' [**] Done show pca of measure in %fs' % (time.time() - start))

    # given coeff of pca_vertex_basis, return body shape
    def mapping(self, coeff):
        coeff = numpy.array(coeff[:self.demo_num, :])
        coeff.shape = (self.demo_num, 1)
        coeff *= self.current_body.m_pca_std
        coeff += self.current_body.m_pca_mean
        coeff = self.M.dot()
        [v, n, f] = self.current_body.v_synthesize(coeff)
        return [v, n, f]
