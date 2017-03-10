#!/usr/bin/python
#coding=utf-8

import sys
sys.path.append("..")
from dataProcess.rawData import *
import numpy.matlib
import scipy.sparse.linalg
import scipy.sparse

class basisData:
    '''
        A basisData contains data of deform-based
        v_basis, v_sigma, v_pca_mean, v_pca_std, v_coeff
        v-synthesize
        d_inv_mean, deform(o, t, std, mean)
        d_basis, d_sigma, d_pca_mean, d_pca_std, d_coeff
        A, LU for transfer deform to vertex
    '''
    __metaclass__ = Singleton
    def __init__(self, data):
        self.data = data
        self.NPYpath = self.data.paras['dataPath']+"NPYdata/"
        self.basisDataPath = self.data.paras['dataPath'] + "NPYdata/basisData/"
        self.d_basis_num = self.data.paras['d_basis_num']
        self.v_basis_num = self.data.paras['v_basis_num']
        [self.v_basis, self.v_sigma, self.v_pca_mean, self.v_pca_std, self.v_coeff] = \
            self.get_vertex_basis()
        [self.d_inv_mean, self.t_deform, self.o_deform, self.mean_deform, self.std_deform] = \
            self.load_deform_data()
        [self.d_basis, self.d_sigma, self.d_pca_mean, self.d_pca_std, self.d_coeff] = \
            self.get_deform_basis()
        [self.A, self.lu] = self.load_d2v_matrix()

    # -------------------------------------------------------
    '''calculating vertex-based presentation(PCA)'''
    # -------------------------------------------------------
    def get_vertex_basis(self):
        print (" [**] begin get_vertex_basis ...")
        v_basis_file = self.basisDataPath + 'v_basis.npy'
        v_sigma_file = self.basisDataPath + 'v_sigma.npy'
        v_pca_mean_file = self.basisDataPath + 'v_pca_mean.npy'
        v_pca_std_file = self.basisDataPath + 'v_pca_std.npy'
        v_coeff_file = self.basisDataPath + 'v_coeff.npy'
        start = time.time()
        if self.data.paras['reload_vertex_basis']:
            # principle component analysis
            v_basis, v_sigma, V = np.linalg.svd(self.data.t_vertex, full_matrices=0)
            v_coeff = np.dot(v_basis.transpose(), self.data.t_vertex)
            v_pca_mean = np.array(np.mean(v_coeff, axis=1))
            v_pca_mean.shape = (v_pca_mean.size, 1)
            v_pca_std = np.array(np.std(v_coeff, axis=1))
            v_pca_std.shape = (v_pca_std.size, 1)
            v_basis = np.array(v_basis[::, :50]).reshape(v_basis.shape[0], 50)
            v_coeff = np.array(v_coeff[:50, ::]).reshape(50, self.data.body_count)
            self.data.save_NPY([v_basis_file, v_sigma_file, v_pca_mean_file, v_pca_std_file, v_coeff_file],
                               [v_basis, v_sigma, v_pca_mean, v_pca_std, v_coeff])
            print (' [**] finish get_vertex_basis in %fs' % (time.time() - start))
            return [v_basis, v_sigma, v_pca_mean, v_pca_std, v_coeff]
        else:
            print (' [**] finish get_vertex_basis in %fs' % (time.time() - start))
            return self.data.load_NPY([v_basis_file, v_sigma_file, v_pca_mean_file, v_pca_std_file, v_coeff_file])

    # --------------------------------------------------------------------------------------------------
    '''synthesize a body by vertex-based, given coeff of pca basis before trunck '''
    # --------------------------------------------------------------------------------------------------
    def v_synthesize(self, weight):
        # start = time.time()
        basis = np.array(self.v_basis[::, :self.v_basis_num]).reshape(self.v_basis.shape[0],self.v_basis_num)
        weight = np.array(weight).reshape(self.v_basis_num, 1)
        for i in range(0, weight.shape[0]):
            weight[i, 0] = max(weight[i, 0], self.v_pca_mean[i] - 3 * self.v_pca_std[i])
            weight[i, 0] = min(weight[i, 0], self.v_pca_mean[i] + 3 * self.v_pca_std[i])
        vertex = self.data.mean_vertex + np.dot(basis, weight) * self.data.std_vertex
        vertex = vertex.reshape(vertex.size, 1)
        # print '  synthesis by vertex-based global in %f s'%(time.time()-start)
        return [vertex, -self.data.o_normals, self.data.o_faces - 1]

    # -------------------------------------------------------
    '''loading deform-based data'''
    # -------------------------------------------------------
    def load_deform_data(self):
        print (" [**] begin load_deform_data ...")
        d_inv_mean_file = self.basisDataPath + 'd_inv_mean.npy'
        t_deform_file = self.basisDataPath + 't_deform.npy'
        o_deform_file = self.basisDataPath + 'o_deform.npy'
        mean_deform_file = self.basisDataPath + 'mean_deform.npy'
        std_deform_file = self.basisDataPath + 'std_deform.npy'
        start = time.time()
        if self.data.paras['reload_deform_data']:
            d_inv_mean = self.get_inv_mean()
            o_deform = np.matlib.zeros((self.data.face_num * 9, self.data.o_vertex.shape[1]))
            # calculate deformation mat of each body shape
            for i in range(0, self.data.face_num):
                print ('loading deformation of each body: NO. ', i)
                face = [k - 1 for k in self.data.o_faces[3 * i:3 * i + 3, 0]]
                for j in range(0, self.data.o_vertex.shape[1]):
                    v1 = self.data.o_vertex[3 * face[0]: 3 * face[0] + 3, j]
                    v2 = self.data.o_vertex[3 * face[1]: 3 * face[1] + 3, j]
                    v3 = self.data.o_vertex[3 * face[2]: 3 * face[2] + 3, j]
                    Q = numpy.matmul(self.assemble_face(v1, v2, v3), d_inv_mean[i]).reshape((9, 1))
                    o_deform[range(i * 9, i * 9 + 9), j] = np.array([q for q in Q.flat])
            # normalized deformation mat
            mean_deform = np.array(o_deform.mean(axis=1))
            std_deform = np.array(np.std(o_deform, axis=1))
            t_deform = o_deform.copy()
            for i in range(0, t_deform.shape[1]):
                t_deform[:, i] -= mean_deform  # .reshape(data.shape[0])
            for i in range(0, t_deform.shape[0]):
                t_deform[i, :] /= std_deform[i, 0]
            self.data.save_NPY([d_inv_mean_file, t_deform_file, o_deform_file, mean_deform_file, std_deform_file],
                               [d_inv_mean, t_deform, o_deform, mean_deform, std_deform])
            print (' [**] finish load_deform_data in %fs' % (time.time() - start))
            return [d_inv_mean, t_deform, o_deform, mean_deform, std_deform]
        else:
            print (' [**] finish load_deform_data in %fs' % (time.time() - start))
            return self.data.load_NPY([d_inv_mean_file, t_deform_file, o_deform_file, mean_deform_file, std_deform_file])

    # ----------------------------------------------------------
    '''calculating the inverse of mean vertex matrix, v^-1 '''
    # ----------------------------------------------------------
    def get_inv_mean(self):
        print (" [**] begin get_inv_mean ...")
        start = time.time()
        d_inv_mean = np.zeros((self.data.face_num, 3, 3))
        for i in range(0, self.data.face_num):
            f = [j - 1 for j in self.data.o_faces[i * 3:i * 3 + 3, 0]]
            d_inv_mean[i] = self.assemble_face(self.data.mean_vertex[f[0] * 3:f[0] * 3 + 3, 0],
                                                self.data.mean_vertex[f[1] * 3:f[1] * 3 + 3, 0],
                                               self.data.mean_vertex[f[2] * 3:f[2] * 3 + 3, 0])
            d_inv_mean[i] = np.linalg.inv(d_inv_mean[i])
        print (' [**] finish get_inv_mean in %fs' % (time.time() - start))
        return d_inv_mean

    # -------------------------------------------------------------------------
    '''import the 4th point of the triangle, and calculate the deformation '''
    # -------------------------------------------------------------------------
    def assemble_face(self, v1, v2, v3):
        v21 = np.array((v2 - v1))
        v31 = np.array((v3 - v1))
        v41 = np.cross(list(v21.flat), list(v31.flat))
        v41 /= np.sqrt(np.linalg.norm(v41))
        return np.column_stack((v21, np.column_stack((v31, v41))))

    # -------------------------------------------------------
    '''calculating deform-based presentation(PCA)'''
    # -------------------------------------------------------
    def get_deform_basis(self):
        print (" [**] begin get_deform_basis ...")
        d_coeff_file = self.basisDataPath + 'd_coeff.npy'
        d_basis_file = self.basisDataPath + 'd_basis.npy'
        d_sigma_file = self.basisDataPath + 'd_sigma.npy'
        d_pca_mean_file = self.basisDataPath + 'd_pca_mean.npy'
        d_pca_std_file = self.basisDataPath + 'd_pca_std.npy'
        start = time.time()
        if self.data.paras['reload_deform_basis']:
            # principle component analysis
            d_basis, d_sigma, V = np.linalg.svd(self.t_deform, full_matrices=0)
            d_coeff = np.dot(d_basis.transpose(), self.t_deform)
            d_pca_mean = np.array(np.mean(d_coeff, axis=1))
            d_pca_mean.shape = (d_pca_mean.size, 1)
            d_pca_std = np.array(np.std(d_coeff, axis=1))
            d_pca_std.shape = (d_pca_std.size, 1)
            d_basis = np.array(d_basis[::, :50]).reshape(d_basis.shape[0], 50)
            d_coeff = np.array(d_coeff[:50, ::]).reshape(50, self.data.body_count)
            self.data.save_NPY([d_coeff_file, d_basis_file, d_sigma_file, d_pca_mean_file, d_pca_std_file],
                               [d_coeff, d_basis, d_sigma, d_pca_mean, d_pca_std])
            print (' [**] finish get_deform_basis in %fs' % (time.time() - start))
            return [d_basis, d_sigma, d_pca_mean, d_pca_std, d_coeff]
        else:
            print (' [**] finish get_deform_basis in %fs' % (time.time() - start))
            return self.data.load_NPY([d_basis_file, d_sigma_file, d_pca_mean_file, d_pca_std_file, d_coeff_file])

    # --------------------------------------------------------------------------
    '''construct the matrix = v_mean_inv.dot(the matrix consists of 0 -1...) '''
    # --------------------------------------------------------------------------
    def construct_coeff_mat(self, mat):
        tmp = -mat.sum(0)
        return np.row_stack((tmp, mat)).transpose()

    # ----------------------------------------------------------------------------------------
    '''cosntruct the related matrix A to change deformation into vertex using global method'''
    # ----------------------------------------------------------------------------------------
    def load_d2v_matrix(self):
        print (' [**] begin reload A&lu maxtrix')
        start = time.time()
        if self.data.paras['reload_A']:
            data = []
            rowidx = []
            colidx = []
            r = 0
            off = self.data.vertex_num * 3
            shape = (self.data.face_num * 9, (self.data.vertex_num + self.data.face_num) * 3)
            # shape = (self.data.face_num * 9+3, (self.data.vertex_num + self.data.face_num) * 3)
            for i in range(0, self.data.face_num):
                print (i)
                coeff = self.construct_coeff_mat(self.d_inv_mean[i])
                face = [c - 1 for c in self.data.o_faces[3 * i:3 * i + 3, 0]]
                v1 = range(face[0] * 3, face[0] * 3 + 3)
                v2 = range(face[1] * 3, face[1] * 3 + 3)
                v3 = range(face[2] * 3, face[2] * 3 + 3)
                v4 = range(off + i * 3, off + i * 3 + 3)
                for j in range(0, 3):
                    data += [c for c in coeff.flat]
                    rowidx += [r, r, r, r, r + 1, r + 1, r + 1, r + 1, r + 2, r + 2, r + 2, r + 2]
                    colidx += [v1[j], v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j]]
                    r += 3
            # # plus 3 rows for x, y, z, their mean is zero
            # # x
            # data += [1 for c in range(0, self.data.vertex_num)]
            # rowidx += [r for c in range(0, self.data.vertex_num)]
            # colidx += [3*c for c in range(0, self.data.vertex_num)]
            # # y
            # data += [1 for c in range(0, self.data.vertex_num)]
            # rowidx += [r+1 for c in range(0, self.data.vertex_num)]
            # colidx += [3*c+1 for c in range(0, self.data.vertex_num)]
            # # z
            # data += [1 for c in range(0, self.data.vertex_num)]
            # rowidx += [r+2 for c in range(0, self.data.vertex_num)]
            # colidx += [3*c+2 for c in range(0, self.data.vertex_num)]

            A = scipy.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
            np.savez(self.NPYpath + "A", row=A.row, col=A.col, data=A.data, shape=A.shape)
            print ('finised A')
        else:
            loader = np.load(self.NPYpath + "A.npz")
            A = scipy.sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
        lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
        # lu = 0
        print (' [**] finish load A&lu matrix in %fs.' % (time.time() - start))
        return [A, lu]

    # --------------------------------------------------------------------------------
    '''synthesize a body by transformation-based, given deformation, output vertex '''
    # -------------------------------------------------------------------------------
    def d_synthesize(self, deformation):
        # tmp = np.array([[0.0], [0.0], [0.0]])
        # deformation = np.row_stack((deformation, tmp))
        Atd = self.A.transpose().dot(deformation)
        x = self.lu.solve(Atd)
        # AtA = self.A.transpose().dot(self.A)
        # x = np.array(scipy.sparse.linalg.spsolve(AtA, Atd))
        x = x[:self.data.vertex_num * 3]
        # move to center
        x.shape = (self.data.vertex_num, 3)
        x_mean = np.mean(x, axis=0)
        for i in range(0, self.data.vertex_num):
            x[i,:] -= x_mean
        x.shape = (self.data.vertex_num*3, 1)
        #---------------
        return [x.reshape(x.size, 1), -self.data.o_normals, self.data.o_faces - 1]

    # -------------------------------------------------------------------------
    '''calculate the corresponding deformation from the input vertex'''
    # -------------------------------------------------------------------------
    def getDeform(self, vertex):
        deform = np.matlib.zeros((9 * self.data.face_num, 1))
        for i in range(0, self.data.face_num):
            face = [k - 1 for k in self.data.o_faces[3 * i:3 * i + 3, 0]]
            v1 = vertex[3 * face[0]: 3 * face[0] + 3, 0]
            v2 = vertex[3 * face[1]: 3 * face[1] + 3, 0]
            v3 = vertex[3 * face[2]: 3 * face[2] + 3, 0]
            Q = numpy.matmul(self.assemble_face(v1, v2, v3), self.d_inv_mean[i]).reshape((9, 1))
            deform[range(i * 9, i * 9 + 9), 0] = np.array([q for q in Q.flat])
        return deform


#######################################################################
#######################################################################
if __name__ == "__main__":
    filename = "../parameter.json"
    data = rawData(filename)
    model = basisData(data)
    for i in range(0, 5):
        deformation = model.o_deform[:,i]
        [x, normals, f] = model.d_synthesize(deformation)
        model.data.save_obj("../../result/test.obj", x, f+1)
        x.shape = (12500, 3)
        print (np.mean(x, axis=0))
        input()