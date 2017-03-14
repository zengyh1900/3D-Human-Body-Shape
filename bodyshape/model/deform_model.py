#!/usr/bin/python
# coding=utf-8

import numpy
import time


# a DeformModel show the PCA space of deform
class DeformModel:

    def __init__(self, male, female):
        self.TYPE = "deform-model"
        self.body = [male, female]
        self.current_body = self.body[0]
        self.paras = self.current_body.paras

        self.demo_num = self.current_body.d_basis_num
        self.ans_path = self.current_body.ans_path + "deform_model/"
        self.deformation = None

    def set_body(self, flag):
        self.current_body = self.body[flag - 1]

    # show all pca of deform-based space
    def show_d_pca(self):
        print(" [**] begin show deform's PCA ...")
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
        print(' [**] finish show pca of deform in %fs' % (time.time() - start))

    # given coeff of pca_deform_basis, return body shape
    def mapping(self, coeff):
        coeff = numpy.array(coeff[:self.demo_num, :])
        coeff.shape = (self.demo_num, 1)
        coeff *= self.current_body.d_pca_std
        coeff += self.current_body.d_pca_mean
        basis = self.current_body.d_basis[:, :self.demo_num]
        d = numpy.matmul(basis, coeff)
        [v, n, f] = self.current_body.d_synthesize(d)
        return [v, n, f]
