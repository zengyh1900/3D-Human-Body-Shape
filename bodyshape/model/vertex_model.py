#!/usr/bin/python
# coding=utf-8

import numpy
import time


# a VertexModel show the PCA space of vertex
class VertexModel:

    def __init__(self, male, female):
        self.TYPE = "vertex-model"
        self.body = [male, female]
        self.current_body = self.body[0]

        self.v_basis_num = self.current_body.paras["v_basis_num"]
        self.ans_path = self.current_body.ans_path + "vertex_model/"
        self.demo_num = self.v_basis_num
        self.deformation = None

    def set_body(self, flag):
        self.current_body = self.body[flag - 1]

    # show all pca of vertex-based space
    def show_v_pca(self):
        print(" [**] begin show vertex's PCA ...")
        start = time.time()
        names = [self.ans_path + "01/", self.ans_path + "02/"]
        for i in range(0, 2):
            self.set_body(i + 1)
            for j in range(0, self.v_basis_num):
                for sign in [-1, +1]:
                    alpha = numpy.zeros((self.v_basis_num, 1))
                    alpha[j] = 3 * sign
                    [v, n, f] = self.current_body.v_synthesize(alpha)
                    fname = names[i] + ('0%d_PC%d_%dsigma.obj' %
                                        (self.current_body.flag_, j, 3 * sign))
                    self.current_body.save_obj(fname, v, f + 1)
        print(' [**] finish show pca of vertex in %fs' % (time.time() - start))

    # given coeff of pca_vertex_basis, return body shape
    def mapping(self, coeff):
        coeff = numpy.array(coeff[:self.demo_num, :]).reshape(self.demo_num, 1)
        coeff *= self.current_body.v_pca_std[:self.demo_num, :]
        coeff += self.current_body.v_pca_mean[:self.demo_num, :]
        [v, n, f] = self.current_body.v_synthesize(coeff)
        return [v, n, f]
