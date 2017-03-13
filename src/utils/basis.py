#!/usr/bin/python
# coding=utf-8

from meta import *
import numpy.matlib
import scipy.sparse.linalg
import scipy.sparse


# A Reshaper contains data of deform-based
# v_basis, v_coeff, v_pca_mean, v_pca_std
# v-synthesize
# d_inv_mean, deform(o, t, std, mean)
# d_basis, d_coeff, d_pca_mean, d_pca_std
# A, LU for transfer deform to vertex
class Reshaper:

    def __init__(self, data):
        self.data = data
        self.flag_ = self.data.flag_
        self.data_path = self.data.paras['data_path'] + "basis/"
        self.d_basis_num = self.data.paras['d_basis_num']
        self.v_basis_num = self.data.paras['v_basis_num']

        [self.v_basis, self.v_coeff, self.v_pca_mean, self.v_pca_std] = \
            self.get_v_basis()
        [self.d_inv_mean, self.deform] = self.load_d_data()
        [self.d_basis, self.d_coeff, self.d_pca_mean, self.d_pca_std] = \
            self.get_d_basis()
        [self.A, self.lu] = self.load_d2v_matrix()

    # calculating vertex-based presentation(PCA)
    def get_v_basis(self):
        print(" [**] begin get_v_basis ...")
        v_basis_file = self.data_path + 'v_basis_0%d.npy' % self.flag_
        v_coeff_file = self.data_path + 'v_coeff_0%d.npy' % self.flag_
        v_pca_mean_file = self.data_path + 'v_pca_mean_0%d.npy' % self.flag_
        v_pca_std_file = self.data_path + 'v_pca_std_0%d.npy' % self.flag_
        start = time.time()
        if self.data.paras['reload_v_basis']:
            # principle component analysis
            v = self.data.vertex.copy()
            v.shape = (self.data.body_num, 3 * self.v_num)
            v = v.transpose()
            v_basis, v_sigma, V = np.linalg.svd(v, full_matrices=0)
            v_coeff = np.dot(v_basis.transpose(), v)
            v_pca_mean = np.array(np.mean(v_coeff, axis=1))
            v_pca_mean.shape = (v_pca_mean.size, 1)
            v_pca_std = np.array(np.std(v_coeff, axis=1))
            v_pca_std.shape = (v_pca_std.size, 1)
            v_basis = np.array(v_basis[::, :50])
            v_basis.shape = (v_basis.shape[0], 50)
            v_coeff = np.array(v_coeff[:50, ::])
            v_coeff.shape = (50, self.data.body_num)
            numpy.save(open(v_basis_file, "wb"), v_basis)
            numpy.save(open(v_coeff_file, "wb"), v_coeff)
            numpy.save(open(v_pca_mean_file, "wb"), v_pca_mean)
            numpy.save(open(v_pca_std_file, "wb"), v_pca_std)
        else:
            v_basis = numpy.load(open(v_basis_file, "rb"))
            v_coeff = numpy.load(open(v_coeff_file, "rb"))
            v_pca_mean = numpy.load(open(v_pca_mean_file, "rb"))
            v_pca_std = numpy.load(open(v_pca_std_file, "rb"))
        print(' [**] finish get_v_basis in %fs' % (time.time() - start))
        return [v_basis, v_coeff, v_pca_mean, v_pca_std]

    # synthesize a body by vertex-based
    def v_synthesize(self, weight):
        basis = np.array(self.v_basis[::, :self.v_basis_num]).reshape(
            self.v_basis.shape[0], self.v_basis_num)
        weight = np.array(weight).reshape(self.v_basis_num, 1)
        v = self.data.mean_vertex + \
            np.dot(basis, weight) * self.data.std_vertex
        v.shape(v.size, 1)
        return [v, -self.data.normals, self.data.facet - 1]

    # loading deform-based data
    def load_d_data(self):
        print(" [**] begin load_d_data ...")
        d_inv_mean_file = self.data_path + 'd_inv_mean_0%d.npy' % self.flag_
        deform_file = self.data_path + 'deform_0%d.npy' % self.flag_
        start = time.time()
        if self.data.paras['reload_d_data']:
            d_inv_mean = self.get_inv_mean()
            deform = np.zeros((self.data.body_num, self.data.f_num, 9))
            # calculate deformation mat of each body shape
            for i in range(0, self.data.face_num):
                print('loading deformation of each body: NO. ', i)
                v = [k - 1 for k in self.data.facet[i, :]]
                for j in range(0, self.data.body_num):
                    v1 = self.data.vertex[j, v[0], :]
                    v2 = self.data.vertex[j, v[1], :]
                    v3 = self.data.vertex[j, v[2], :]
                    Q = self.assemble_face(v1, v2, v3).dot(d_inv_mean[i])
                    Q.shape = (9, 1)
                    deform[j, i, :] = Q.flat
            numpy.save(open(d_inv_mean_file, "wb"), d_inv_mean)
            numpy.save(open(deform_file, "wb"), deform)
        else:
            d_inv_mean = numpy.load(open(d_inv_mean_file, "rb"))
            deform = numpy.load(open(deform_file, "rb"))
        print(' [**] finish load_d_data in %fs' % (time.time() - start))
        return[d_inv_mean, deform]

    # calculating the inverse of mean vertex matrix, v^-1
    def get_inv_mean(self):
        print(" [**] begin get_inv_mean ...")
        start = time.time()
        d_inv_mean = np.zeros((self.data.face_num, 3, 3))
        for i in range(0, self.data.face_num):
            v = [j - 1 for j in self.data.facet[i, :]]
            v1 = self.data.mean_vertex[v[0], :]
            v2 = self.data.mean_vertex[v[1], :]
            v3 = self.data.mean_vertex[v[2], :]
            d_inv_mean[i] = self.assemble_face(v1, v2, v3)
            d_inv_mean[i] = np.linalg.inv(d_inv_mean[i])
        print(' [**] finish get_inv_mean in %fs' % (time.time() - start))
        return d_inv_mean

    # import the 4th point of the triangle, and calculate the deformation
    def assemble_face(self, v1, v2, v3):
        v21 = np.array((v2 - v1))
        v31 = np.array((v3 - v1))
        v41 = np.cross(list(v21.flat), list(v31.flat))
        v41 /= np.sqrt(np.linalg.norm(v41))
        return np.column_stack((v21, np.column_stack((v31, v41))))

    # calculating deform-based presentation(PCA)
    def get_d_basis(self):
        print(" [**] begin get_d_basis ...")
        d_basis_file = self.data_path + 'd_basis_0%d.npy' % self.flag_
        d_coeff_file = self.data_path + 'd_coeff_0%d.npy' % self.flag_
        d_pca_mean_file = self.data_path + 'd_pca_mean_0%d.npy' % self.flag_
        d_pca_std_file = self.data_path + 'd_pca_std_0%d.npy' % self.flag_
        start = time.time()
        if self.data.paras['reload_d_basis']:
            # principle component analysis
            deform = self.data.deform.copy()
            deform.shape = (9 * self.data.f_num, self.data.body_num)
            deform = deform.transpose()
            d_basis, d_sigma, V = np.linalg.svd(deform, full_matrices=0)
            d_coeff = np.dot(d_basis.transpose(), deform)
            d_pca_mean = np.array(np.mean(d_coeff, axis=1))
            d_pca_mean.shape = (d_pca_mean.size, 1)
            d_pca_std = np.array(np.std(d_coeff, axis=1))
            d_pca_std.shape = (d_pca_std.size, 1)
            d_basis = np.array(d_basis[::, :50])
            d_basis.shape = (d_basis.shape[0], 50)
            d_coeff = np.array(d_coeff[:50, ::])
            d_coeff.shape = (50, self.data.body_num)
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
        return np.row_stack((tmp, mat)).transpose()

    # cosntruct the related matrix A to change deformation into vertex using
    # global method
    def load_d2v_matrix(self):
        print(' [**] begin reload A&lu maxtrix')
        start = time.time()
        if self.data.paras['reload_A']:
            data = []
            rowidx = []
            colidx = []
            r = 0
            off = self.data.v_num * 3
            shape = (self.data.f_num * 9,
                     (self.data.v_num + self.data.f_num) * 3)
            for i in range(0, self.data.f_num):
                print(i)
                coeff = self.construct_coeff_mat(self.d_inv_mean[i])
                face = [c - 1 for c in self.data.o_faces[3 * i:3 * i + 3, 0]]
                v1 = range(face[0] * 3, face[0] * 3 + 3)
                v2 = range(face[1] * 3, face[1] * 3 + 3)
                v3 = range(face[2] * 3, face[2] * 3 + 3)
                v4 = range(off + i * 3, off + i * 3 + 3)
                for j in range(0, 3):
                    data += [c for c in coeff.flat]
                    rowidx += [r, r, r, r, r + 1, r + 1, r +
                               1, r + 1, r + 2, r + 2, r + 2, r + 2]
                    colidx += [v1[j], v2[j], v3[j], v4[j], v1[j],
                               v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j]]
                    r += 3

            A = scipy.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
            np.savez(self.NPYpath + "A", row=A.row,
                     col=A.col, data=A.data, shape=A.shape)
            print('finised A')
        else:
            loader = np.load(self.NPYpath + "A.npz")
            A = scipy.sparse.coo_matrix(
                (loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
        lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
        # lu = 0
        print(' [**] finish load A&lu matrix in %fs.' % (time.time() - start))
        return [A, lu]

    # synthesize a body by deform-based, given deformation, output vertex
    def d_synthesize(self, deformation):
        Atd = self.A.transpose().dot(deformation)
        x = self.lu.solve(Atd)
        x = x[:self.data.vertex_num * 3]
        # move to center
        x.shape = (self.data.vertex_num, 3)
        x_mean = np.mean(x, axis=0)
        for i in range(0, self.data.vertex_num):
            x[i, :] -= x_mean
        x.shape = (self.data.vertex_num * 3, 1)
        return [x.reshape(x.size, 1), -self.data.o_normals, self.data.o_faces - 1]

    # calculate the corresponding deformation from the input vertex
    def getDeform(self, vertex):
        deform = np.matlib.zeros((9 * self.data.face_num, 1))
        for i in range(0, self.data.face_num):
            face = [k - 1 for k in self.data.o_faces[3 * i:3 * i + 3, 0]]
            v1 = vertex[3 * face[0]: 3 * face[0] + 3, 0]
            v2 = vertex[3 * face[1]: 3 * face[1] + 3, 0]
            v3 = vertex[3 * face[2]: 3 * face[2] + 3, 0]
            Q = numpy.matmul(self.assemble_face(v1, v2, v3),
                             self.d_inv_mean[i]).reshape((9, 1))
            deform[range(i * 9, i * 9 + 9), 0] = np.array([q for q in Q.flat])
        return deform


# test for this module
if __name__ == "__main__":
    male = MetaData("../parameter.json", 1)
    male_reshaper = Reshaper(male)

    female = MetaData("../parameter.json", 2)
    female_reshaper = Reshaper(female)
    # num = numpy.array([i for i in range(0, 24)]).reshape(2, 3, 4)
    # print(num, num.shape)
    # num.shape = (2, 12)
    # num = num.transpose()
    # print(num, num.shape)
