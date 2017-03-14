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
        self.current_body = self.body[0]
        self.paras = self.current_body.paras

        self.demo_num = self.current_body.m_basis_num
        self.ans_path = self.current_body.ans_path + "measure_model/"
        self.data_path = self.paras["data_path"]
        self.deformation = None

        self.m_ = self.M2V()

    # set current body
    def set_body(self, flag):
        self.current_body = self.body[flag - 1]

    # calculate the mapping matrix from measures to vertex-based
    def M2V(self):
        print(' [**] begin load M2V matrix ... ')
        start = time.time()
        m2v = []
        names = [self.data_path + "M2V_01.npy", self.data_path + "M2V_02.npy"]
        if self.current_body.paras["reload_M"]:
            for i, body in enumerate(self.body):
                V = numpy.array(body.v_coeff.transpose().copy())
                V.shape = (body.v_coeff.size, 1)
                M = body.build_equation(body.m_coeff, body.v_basis_num)
                # solve transform matrix
                MtM = M.transpose().dot(M)
                MtV = M.transpose().dot(V)
                ans = numpy.array(scipy.sparse.linalg.spsolve(MtM, MtV))
                ans.shape = (body.v_basis_num, body.m_basis_num)
                m2v.append(ans)
                numpy.save(open(names[i], "wb"), ans)
        else:
            for fname in names:
                m2v.append(numpy.load(open(fname, "rb")))
        print("  finish load M2V matrix in %fs" % (time.time() - start))
        return m2v

    # show all pca of vertex-based space
    def show_m_pca(self):
        print(" [**] begin show measure's PCA ...")
        start = time.time()
        names = [self.ans_path + "01/", self.ans_path + "02/"]
        for i in range(0, 2):
            self.set_body(i + 1)
            for j in range(0, self.demo_num):
                for sign in [-1, +1]:
                    alpha = numpy.zeros((self.demo_num, 1))
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
        m2v = self.m_[self.current_body.flag_ - 1]
        coeff = m2v.dot(coeff)
        [v, n, f] = self.current_body.v_synthesize(coeff)
        return [v, n, f]
