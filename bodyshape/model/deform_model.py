#!/usr/bin/python
# coding=utf-8

import numpy
import time


# a DeformModel show the PCA space of deform
class DeformModel:

    def __init__(self, data):
        self.TYPE = "deform-model"
        self.data = data
        self.d_basis_num = self.data.paras["d_basis_num"]
        self.demo_num = self.d_basis_num
        self.deformation = None

    # show deformation-based synthesize(PCA)
    def show_d_pca(self):
        print(" [**] begin show deformation's PCA ...")
        start = time.time()
        for id in range(0, self.d_basis_num):
            for sign in [-1, +1]:
                alpha = numpy.zeros((self.d_basis_num, 1))
                alpha[id] = 3 * sign
                [vertex, n, f] = self.mapping(alpha)
                fname = self.data.ans_path + \
                    ('0%d_PC%d_%dsigma.obj' % (self.data.flag_, id, 3 * sign))
                self.data.save_obj(fname, vertex, self.data.o_faces)
        print(' [**] finish show pca of deform in %fs' % (time.time() - start))

    # given coeff of pca deform_basis, return body shape
    def mapping(self, weight):
        weight = numpy.array(weight[:self.demo_num, :]).reshape(self.demo_num, 1)
        weight = weight * self.data.d_pca_std[:self.demo_num, :]
        weight += self.data.d_pca_mean[:self.demo_num, :]
        basis = self.data.d_basis[:, :self.d_basis_num]
        deformation = numpy.matmul(basis, weight) * self.data.std_deform
        deformation += self.data.mean_deform
        [v, n, f] = self.data.d_synthesize(deformation)
        return [v, n, f]
