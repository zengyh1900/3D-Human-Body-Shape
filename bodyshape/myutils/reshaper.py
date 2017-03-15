#!/usr/bin/python
# coding=utf-8

from .meta import *
from .miner import *


# A Reshaper contains meta
# v_basis, v_coeff, v_pca_mean, v_pca_std
# v-synthesize
# d_inv_mean, deform(o, t, std, mean)
# d_basis, d_coeff, d_pca_mean, d_pca_std
# A, LU for transfer deform to vertex
class Reshaper:

    def __init__(self, data):
        self.data = data
        self.paras = data.paras
        self.flag_ = data.flag_
        self.ans_path = data.ans_path
        self.data_path = self.paras["data_path"] + "reshaper/"

        self.v_basis_num = self.paras["v_basis_num"]
        self.d_basis_num = self.paras["d_basis_num"]
        self.m_basis_num = self.paras["m_basis_num"]

        [self.measure_str, self.cp] = [data.measure_str, data.cp]
        [self.part, self.mask] = [data.part, data.mask]
        [self.v_num, self.f_num, self.m_num, self.p_num, self.body_num] = \
            [data.v_num, data.f_num, data.m_num, data.p_num, data.body_num]
        [self.facet, self.file_list, self.normals] = \
            [data.facet, data.file_list, data.normals]
        [self.vertex, self.mean_vertex, self.std_vertex] = \
            [data.vertex, data.mean_vertex, data.std_vertex]
        [self.d_inv_mean, self.deform] = \
            [data.d_inv_mean, data.deform]
        [self.measure, self.mean_measure, self.std_measure, self.t_measure] = \
            [data.measure, data.mean_measure, data.std_measure, data.t_measure]

        [self.calc_measure, self.build_equation, self.get_deform] = \
            [data.calc_measure, data.build_equation, data.get_deform]
        self.save_obj = data.save_obj

        [self.v_basis, self.v_coeff, self.v_pca_mean, self.v_pca_std] = \
            self.get_v_basis()
        [self.d_basis, self.d_coeff, self.d_pca_mean, self.d_pca_std] = \
            self.get_d_basis()
        [self.m_basis, self.m_coeff, self.m_pca_mean, self.m_pca_std] = \
            self.get_m_basis()
        [self.d2v_, self.lu] = self.load_d2v_matrix()

    # calculating vertex-based presentation(PCA)
    def get_v_basis(self):
        print(" [**] begin get_v_basis ...")
        v_basis_file = self.data_path + 'v_basis_0%d.npy' % self.flag_
        start = time.time()
        v = self.vertex
        v.shape = (self.body_num, 3 * self.v_num)
        v = v.transpose()
        if self.paras['reload_v_basis']:
            # principle component analysis
            v_basis, v_sigma, V = numpy.linalg.svd(v, full_matrices=0)
            v_basis = numpy.array(v_basis[:, :50]).reshape(3 * self.v_num, 50)
            numpy.save(open(v_basis_file, "wb"), v_basis)
        else:
            v_basis = numpy.load(open(v_basis_file, "rb"))
        v_coeff = numpy.dot(v_basis.transpose(), v)
        v_coeff = numpy.array(v_coeff[:self.v_basis_num, :])
        v_coeff.shape = (self.v_basis_num, self.body_num)
        v_pca_mean = numpy.array(numpy.mean(v_coeff, axis=1))
        v_pca_mean.shape = (v_pca_mean.size, 1)
        v_pca_std = numpy.array(numpy.std(v_coeff, axis=1))
        v_pca_std.shape = (v_pca_std.size, 1)
        v = v.transpose()
        v.shape = (self.body_num, self.v_num, 3)
        print(' [**] finish get_v_basis in %fs' % (time.time() - start))
        return [v_basis, v_coeff, v_pca_mean, v_pca_std]

    # calculating deform-based presentation(PCA)
    def get_d_basis(self):
        print(" [**] begin get_d_basis ...")
        d_basis_file = self.data_path + 'd_basis_0%d.npy' % self.flag_
        start = time.time()
        d = self.deform
        d.shape = (self.body_num, 9 * self.f_num)
        d = d.transpose()
        if self.paras['reload_d_basis']:
            # principle component analysis
            d_basis, d_sigma, V = numpy.linalg.svd(d, full_matrices=0)
            d_basis = numpy.array(d_basis[:, :50]).reshape(9 * self.f_num, 50)
            numpy.save(open(d_basis_file, "wb"), d_basis)
        else:
            d_basis = numpy.load(open(d_basis_file, "rb"))
        d_coeff = numpy.dot(d_basis.transpose(), d)
        d_coeff = numpy.array(d_coeff[:self.d_basis_num, :])
        d_coeff.shape = (self.d_basis_num, self.body_num)
        d_pca_mean = numpy.array(numpy.mean(d_coeff, axis=1))
        d_pca_mean.shape = (d_pca_mean.size, 1)
        d_pca_std = numpy.array(numpy.std(d_coeff, axis=1))
        d_pca_std.shape = (d_pca_std.size, 1)
        d = d.transpose()
        d.shape = (self.body_num, self.f_num, 9)
        print(' [**] finish get_d_basis in %fs' % (time.time() - start))
        return [d_basis, d_coeff, d_pca_mean, d_pca_std]

    # calculating measure-based presentation(PCA)
    def get_m_basis(self):
        print(" [**] begin get_measure_basis ...")
        m_basis_file = self.data_path + 'm_basis_0%d.npy' % self.flag_
        start = time.time()
        if self.data.paras["reload_m_basis"]:
            # principle component analysis
            m_basis, g, M = numpy.linalg.svd(self.measure, full_matrices=0)
            numpy.save(open(m_basis_file, "wb"), m_basis)
        else:
            m_basis = numpy.load(open(m_basis_file, "rb"))
        m_coeff = numpy.dot(m_basis.transpose(), self.measure)
        m_coeff = numpy.array(m_coeff[:self.m_basis_num, :])
        m_coeff.shape = (self.m_basis_num, self.body_num)
        m_pca_mean = numpy.array(numpy.mean(m_coeff, axis=1))
        m_pca_mean.shape = (m_pca_mean.size, 1)
        m_pca_std = numpy.array(numpy.std(m_coeff, axis=1))
        m_pca_std.shape = (m_pca_std.size, 1)
        print(' [**] finish get_m_basis in %fs' % (time.time() - start))
        return [m_basis, m_coeff, m_pca_mean, m_pca_std]

    # cosntruct the related matrix A to change deformation into vertex using
    # global method
    def load_d2v_matrix(self):
        print(' [**] begin reload A&lu maxtrix')
        start = time.time()
        d2v_file = self.data_path + "d2v_0%d" % self.flag_
        if self.paras['reload_d2v']:
            data = []
            rowidx = []
            colidx = []
            r = 0
            off = self.v_num * 3
            shape = (self.f_num * 9, (self.v_num + self.f_num) * 3)
            for i in range(0, self.f_num):
                coeff = self.construct_coeff_mat(self.d_inv_mean[i])
                v = [c - 1 for c in self.facet[i, :]]
                v1 = range(v[0] * 3, v[0] * 3 + 3)
                v2 = range(v[1] * 3, v[1] * 3 + 3)
                v3 = range(v[2] * 3, v[2] * 3 + 3)
                v4 = range(off + i * 3, off + i * 3 + 3)
                for j in range(0, 3):
                    data += [c for c in coeff.flat]
                    rowidx += [r, r, r, r, r + 1, r + 1, r + 1,
                               r + 1, r + 2, r + 2, r + 2, r + 2]
                    colidx += [v1[j], v2[j], v3[j], v4[j], v1[j],
                               v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j]]
                    r += 3
            d2v = scipy.sparse.coo_matrix(
                (data, (rowidx, colidx)), shape=shape)
            numpy.savez(d2v_file, row=d2v.row, col=d2v.col,
                        data=d2v.data, shape=d2v.shape)
            print('finised d2v')
        else:
            loader = numpy.load(d2v_file + ".npz")
            d2v = scipy.sparse.coo_matrix(
                (loader['data'], (loader['row'], loader['col'])),
                shape=loader['shape'])
        lu = scipy.sparse.linalg.splu(d2v.transpose().dot(d2v).tocsc())
        print(' [**] finish load A&lu matrix in %fs.' % (time.time() - start))
        return [d2v, lu]

    # synthesize a body by vertex-based
    def v_synthesize(self, weight):
        basis = numpy.array(self.v_basis[::, :self.v_basis_num])
        basis.shape = (3 * self.v_num, self.v_basis_num)
        v = numpy.dot(basis, weight)
        v.shape = (self.v_num, 3)
        return [v, -self.normals, self.facet - 1]

    # construct the matrix = v_mean_inv.dot(the matrix consists of 0 -1...)
    def construct_coeff_mat(self, mat):
        tmp = -mat.sum(0)
        return numpy.row_stack((tmp, mat)).transpose()

    # synthesize a body by deform-based, given deformation, output vertex
    def d_synthesize(self, deformation):
        d = numpy.array(deformation.flat).reshape(deformation.size, 1)
        Atd = self.d2v_.transpose().dot(d)
        x = self.lu.solve(Atd)
        x = x[:self.v_num * 3]
        # move to center
        x.shape = (self.v_num, 3)
        x_mean = numpy.mean(x, axis=0)
        x -= x_mean
        return [x, -self.normals, self.facet - 1]
