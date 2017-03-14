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

        [self.v_basis, self.v_coeff, self.v_pca_mean, self.v_pca_std] = \
            self.get_v_basis()
        [self.d_basis, self.d_coeff, self.d_pca_mean, self.d_pca_std] = \
            self.get_d_basis()

        [self.calc_measures, self.build_equation, self.get_deform] = \
            [data.calc_measures, data.build_equation, data.get_deform]
        self.save_obj = data.save_obj

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
            print(v_basis.shape)
            v_basis = numpy.array(v_basis[:, 50]).reshape(3 * self.v_num, 50)
            numpy.save(open(v_basis_file, "wb"), v_basis)
        else:
            v_basis = numpy.load(open(v_basis_file, "rb"))
        v_coeff = numpy.dot(v_basis.transpose(), v)
        v_pca_mean = numpy.array(numpy.mean(v_coeff, axis=1))
        v_pca_mean.shape = (v_pca_mean.size, 1)
        v_pca_std = numpy.array(numpy.std(v_coeff, axis=1))
        v_pca_std.shape = (v_pca_std.size, 1)
        print(v_coeff.shape)
        input()

        v = v.transpose()
        v.shape = (self.body_num, self.v_num, 3)
        print(' [**] finish get_v_basis in %fs' % (time.time() - start))
        return [v_basis, v_coeff, v_pca_mean, v_pca_std]

    # synthesize a body by vertex-based
    def v_synthesize(self, weight):
        basis = numpy.array(self.v_basis[::, :self.v_basis_num])
        basis.shape = (self.v_num, self.v_basis_num)
        weight = numpy.array(weight).reshape(self.v_basis_num, 1)
        v = numpy.dot(basis, weight)
        v.shape = (self.v_num, 3)
        return [v, -self.normals, self.facet - 1]

    # calculating deform-based presentation(PCA)
    def get_d_basis(self):
        print(" [**] begin get_d_basis ...")
        d_basis_file = self.data_path + 'd_basis_0%d.numpy.' % self.flag_
        d_coeff_file = self.data_path + 'd_coeff_0%d.numpy.' % self.flag_
        d_pca_mean_file = self.data_path + 'd_pca_mean_0%d.numpy.' % self.flag_
        d_pca_std_file = self.data_path + 'd_pca_std_0%d.numpy.' % self.flag_
        start = time.time()
        if self.paras['reload_d_basis']:
            # principle component analysis
            deform = self.deform.copy()
            deform.shape = (9 * self.f_num, self.body_num)
            deform = deform.transpose()
            d_basis, d_sigma, V = numpy.linalg.svd(deform, full_matrices=0)
            d_coeff = numpy.dot(d_basis.transpose(), deform)
            d_pca_mean = numpy.array(numpy.mean(d_coeff, axis=1))
            d_pca_mean.shape = (d_pca_mean.size, 1)
            d_pca_std = numpy.array(numpy.std(d_coeff, axis=1))
            d_pca_std.shape = (d_pca_std.size, 1)
            d_basis = numpy.array(d_basis[::, :50])
            d_basis.shape = (d_basis.shape[0], 50)
            d_coeff = numpy.array(d_coeff[:50, ::])
            print(d_coeff.shape)
            d_coeff.shape = (50, self.body_num)
            numpy.save(open(d_basis_file, "wb"), d_basis)
            numpy.save(open(d_coeff_file, "wb"), d_coeff)
            numpy.save(open(d_pca_mean_file, "wb"), d_pca_mean)
            numpy.save(open(d_pca_std_file, "wb"), d_pca_std)
        else:
            d_basis = numpy.load(open(d_basis_file, "rb"))
            d_coeff = numpy.load(open(d_coeff_file, "rb"))
            d_pca_mean = numpy.load(open(d_pca_mean_file, "rb"))
            d_pca_std = numpy.load(open(d_pca_std_file, "wb"))
        print(' [**] finish get_d_basis in %fs' % (time.time() - start))
        return [d_basis, d_coeff, d_pca_mean, d_pca_std]

    # construct the matrix = v_mean_inv.dot(the matrix consists of 0 -1...)
    def construct_coeff_mat(self, mat):
        tmp = -mat.sum(0)
        return numpy.row_stack((tmp, mat)).transpose()

    # cosntruct the related matrix A to change deformation into vertex using
    # global method
    def load_d2v_matrix(self):
        print(' [**] begin reload A&lu maxtrix')
        start = time.time()
        if self.paras['reload_A']:
            data = []
            rowidx = []
            colidx = []
            r = 0
            off = self.v_num * 3
            shape = (self.f_num * 9, (self.v_num + self.f_num) * 3)
            for i in range(0, self.f_num):
                print(i)
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
            A = scipy.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
            numpy.savez(self.data_path + "A", row=A.row,
                        col=A.col, data=A.data, shape=A.shape)
            print('finised A')
        else:
            loader = numpy.load(self.numpy.path + "A.numpy.")
            A = scipy.sparse.coo_matrix(
                (loader['data'], (loader['row'], loader['col'])),
                shape=loader['shape'])
        lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
        print(' [**] finish load A&lu matrix in %fs.' % (time.time() - start))
        return [A, lu]

    # synthesize a body by deform-based, given deformation, output vertex
    def d_synthesize(self, deformation):
        d = numpy.array(deformation.flat).reshape(deformation.size, 1)
        Atd = self.A.transpose().dot(d)
        x = self.lu.solve(Atd)
        x = x[:self.v_num * 3]
        # move to center
        x.shape = (self.v_num, 3)
        x_mean = numpy.mean(x, axis=0)
        x -= x_mean
        return [x, -self.normals, self.facet - 1]


# test for this module
if __name__ == "__main__":
    filename = "../parameter.json"

    male_data = MetaData(filename, 1)
    male_reshaper = Reshaper(male_data)

    female_data = MetaData(filename, 2)
    female_reshaper = Reshaper(female_data)
