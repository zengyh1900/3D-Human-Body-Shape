#!/usr/bin/python
# coding=utf-8

import sys
sys.path.append("..")
import scipy.sparse.linalg
import scipy.sparse
from dataProcess.rawData import *
from dataProcess.basisData import *
from dataProcess.Masker import *


class dataModel:
    '''
        A dataModel provide basic function and data
    '''
    __metaclass__ = Singleton

    def __init__(self, basic, masker):
        self.data = basic.data
        self.basic = basic
        self.masker = masker
        self.loadData()
        switcher = {
            1: lambda: [0, 0],
            2: self.v_part_dad,
            3: self.all_dav,
            4: self.part_dav,
        }
        [self.m2v_A, self.m2v_lu] = switcher.get(
            self.paras['Hybrid_method'], lambda: [0, 0])()

    # --------------------------------------------------------------------------------
    '''load all data it need '''
    # -------------------------------------------------------------------------------

    def loadData(self):
        self.paras = self.data.paras
        self.NPYpath = self.paras["dataPath"] + "NPYdata/"
        self.ansPath = self.paras['ansPath']
        [self.o_normals, self.v_basis_num, self.d_basis_num] = \
            [self.data.o_normals, self.basic.v_basis_num, self.basic.d_basis_num]
        [self.body_count, self.measure_num, self.vertex_num, self.face_num] = \
            [self.data.body_count, self.data.measure_num,
                self.data.vertex_num, self.data.face_num]
        [self.o_file_list, self.o_faces, self.o_vertex, self.t_vertex, self.mean_vertex, self.std_vertex] = \
            [self.data.o_file_list, self.data.o_faces, self.data.o_vertex,
                self.data.t_vertex, self.data.mean_vertex, self.data.std_vertex]
        [self.o_deform, self.t_deform, self.mean_deform, self.std_deform] = \
            [self.basic.o_deform, self.basic.t_deform,
                self.basic.mean_deform, self.basic.std_deform]
        [self.v_basis, self.v_pca_mean, self.v_pca_std, self.v_coeff] = \
            [self.basic.v_basis, self.basic.v_pca_mean,
                self.basic.v_pca_std, self.basic.v_coeff]
        [self.d_inv_mean, self.d_basis, self.d_pca_mean, self.d_pca_std, self.d_coeff] = \
            [self.basic.d_inv_mean, self.basic.d_basis, self.basic.d_pca_mean,
                self.basic.d_pca_std, self.basic.d_coeff]
        [self.measure_str, self.o_measures, self.t_measures, self.mean_measures, self.std_measures] = \
            [self.data.measure_str, self.data.o_measures, self.data.t_measures,
                self.data.mean_measures, self.data.std_measures]
        [self.build_equation, self.construct_coeff_mat] = [
            self.data.build_equation, self.basic.construct_coeff_mat]
        [self.v_synthesize, self.d_synthesize, self.save_obj, self.calc_measures] = \
            [self.basic.v_synthesize, self.basic.d_synthesize,
                self.data.save_obj, self.data.calc_measures]
        [self.save_NPY, self.load_NPY] = [
            self.data.save_NPY, self.data.load_NPY]
        [self.color_list, self.color_set] = [
            self.masker.color_list, self.masker.color_set]
        [self.p2m, self.p2p, self.p2f, self.m2p, self.m2f, self.m2v, self.mask] = \
            [self.masker.p2m, self.masker.p2p, self.masker.p2f, self.masker.m2p,
                self.masker.m2f, self.masker.m2v, self.masker.mask]
        self.face_vertex = self.masker.face_vertex

    # ----------------------------------------------------------------------------------------
    '''pre compute for Hybrid: part deform + deform'''
    # ----------------------------------------------------------------------------------------

    def v_part_dad(self):
        print(' [**] begin load part_dad mapping matrix')
        start = time.time()
        A_list_file = self.NPYpath + \
            "localMapper/v_part_dad_A_list%d.npy" % self.paras[
                'mapping_version']
        m2v_A = []
        m2v_lu = []
        if self.paras['reload_v_part_dad']:
            for i in range(0, self.measure_num):
                pure_face = self.face_vertex[i][0]
                edge_face = self.face_vertex[i][1]
                vertex_set = self.face_vertex[i][2]
                print('   pure_face: %d, edge_face: %d' %
                      (len(pure_face), len(edge_face)))
                # for vertex of pure faces
                off = len(vertex_set) * 3
                shape = ((len(pure_face) + len(edge_face)) * 9,
                         (len(vertex_set) + len(pure_face) + len(edge_face)) * 3)
                # calculate A and lu for each measures
                data = []
                row = []
                col = []
                r = 0
                # for deform of pure faces
                for j in range(0, len(pure_face)):
                    coeff = self.construct_coeff_mat(
                        self.d_inv_mean[pure_face[j]])
                    f = [c - 1 for c in self.o_faces[3 *
                                                     pure_face[j]:3 * pure_face[j] + 3, 0]]
                    v1 = range(vertex_set.index(
                        f[0]) * 3, vertex_set.index(f[0]) * 3 + 3)
                    v2 = range(vertex_set.index(
                        f[1]) * 3, vertex_set.index(f[1]) * 3 + 3)
                    v3 = range(vertex_set.index(
                        f[2]) * 3, vertex_set.index(f[2]) * 3 + 3)
                    v4 = range(off + j * 3, off + j * 3 + 3)
                    for k in range(0, 3):
                        data += [c for c in coeff.flat]
                        row += [r, r, r, r, r + 1, r + 1, r + 1,
                                r + 1, r + 2, r + 2, r + 2, r + 2]
                        col += [v1[k], v2[k], v3[k], v4[k], v1[k], v2[k],
                                v3[k], v4[k], v1[k], v2[k], v3[k], v4[k]]
                        r += 3
                # for deform of edge faces
                for j in range(0, len(edge_face)):
                    coeff = self.construct_coeff_mat(
                        self.d_inv_mean[edge_face[j]])
                    f = [c - 1 for c in self.o_faces[3 *
                                                     edge_face[j]:3 * edge_face[j] + 3, 0]]
                    v1 = range(vertex_set.index(
                        f[0]) * 3, vertex_set.index(f[0]) * 3 + 3)
                    v2 = range(vertex_set.index(
                        f[1]) * 3, vertex_set.index(f[1]) * 3 + 3)
                    v3 = range(vertex_set.index(
                        f[2]) * 3, vertex_set.index(f[2]) * 3 + 3)
                    v4 = range(off + len(pure_face) * 3 + j * 3,
                               off + len(pure_face) * 3 + j * 3 + 3)
                    for k in range(0, 3):
                        data += [c for c in coeff.flat]
                        row += [r, r, r, r, r + 1, r + 1, r + 1,
                                r + 1, r + 2, r + 2, r + 2, r + 2]
                        col += [v1[k], v2[k], v3[k], v4[k], v1[k], v2[k],
                                v3[k], v4[k], v1[k], v2[k], v3[k], v4[k]]
                        r += 3
                A = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
                lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
                m2v_A.append(A)
                m2v_lu.append(lu)
            np.save(open(A_list_file, 'wb'), m2v_A)
        else:
            m2v_A = np.load(open(A_list_file, "rb"))
            m2v_lu = []
            for i in range(0, len(m2v_A)):
                m2v_lu.append(scipy.sparse.linalg.splu(
                    m2v_A[i].transpose().dot(m2v_A[i]).tocsc()))
        print(' [**] finish loading m2d2v matrix in %fs.' %
              (time.time() - start))
        return [m2v_A, m2v_lu]

    # ----------------------------------------------------------------------------------------
    '''pre compute for Hybrid: part deform + deform'''
    # ----------------------------------------------------------------------------------------

    def part_dad(self):
        print(' [**] begin load part_dad mapping matrix')
        start = time.time()
        A_list_file = self.NPYpath + \
            "localMapper/part_dad_A%d.npy" % self.paras['mapping_version']
        if self.paras['reload_part_dad']:
            m2v_A = []
            m2v_lu = []
            # for each measures
            for i in range(0, self.measure_num):
                print('  processing %d measure...' % i)
                vertex_set = self.m2v[i][0]
                # for vertex of pure faces
                off = len(vertex_set) * 3
                l1 = len(self.m2f[i][0])
                l2 = len(self.m2f[i][1])
                shape = ((l1 + l2) * 9, (len(vertex_set) + l1 + l2) * 3)
                # calculate A and lu for each measures
                data = []
                row = []
                col = []
                r = 0
                # for deform of pure faces
                for j in range(0, len(self.m2f[i][0])):
                    coeff = self.construct_coeff_mat(
                        self.d_inv_mean[self.m2f[i][0][j]])
                    f = [c - 1 for c in self.o_faces[3 * self.m2f[i]
                                                     [0][j]:3 * self.m2f[i][0][j] + 3, 0]]
                    v1 = range(vertex_set.index(
                        f[0]) * 3, vertex_set.index(f[0]) * 3 + 3)
                    v2 = range(vertex_set.index(
                        f[1]) * 3, vertex_set.index(f[1]) * 3 + 3)
                    v3 = range(vertex_set.index(
                        f[2]) * 3, vertex_set.index(f[2]) * 3 + 3)
                    v4 = range(off + j * 3, off + j * 3 + 3)
                    for k in range(0, 3):
                        data += [c for c in coeff.flat]
                        row += [r, r, r, r, r + 1, r + 1, r + 1,
                                r + 1, r + 2, r + 2, r + 2, r + 2]
                        col += [v1[k], v2[k], v3[k], v4[k], v1[k], v2[k],
                                v3[k], v4[k], v1[k], v2[k], v3[k], v4[k]]
                        r += 3
                # for deform of edge faces
                for j in range(0, len(self.m2f[i][1])):
                    coeff = self.construct_coeff_mat(
                        self.d_inv_mean[self.m2f[i][1][j]])
                    f = [c - 1 for c in self.o_faces[3 * self.m2f[i]
                                                     [1][j]:3 * self.m2f[i][1][j] + 3, 0]]
                    v1 = range(vertex_set.index(
                        f[0]) * 3, vertex_set.index(f[0]) * 3 + 3)
                    v2 = range(vertex_set.index(
                        f[1]) * 3, vertex_set.index(f[1]) * 3 + 3)
                    v3 = range(vertex_set.index(
                        f[2]) * 3, vertex_set.index(f[2]) * 3 + 3)
                    v4 = range(off + l1 * 3 + j * 3, off + l1 * 3 + j * 3 + 3)
                    for k in range(0, 3):
                        data += [c for c in coeff.flat]
                        row += [r, r, r, r, r + 1, r + 1, r + 1,
                                r + 1, r + 2, r + 2, r + 2, r + 2]
                        col += [v1[k], v2[k], v3[k], v4[k], v1[k], v2[k],
                                v3[k], v4[k], v1[k], v2[k], v3[k], v4[k]]
                        r += 3
                A = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
                lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
                m2v_A.append(A)
                m2v_lu.append(lu)
            np.save(open(A_list_file, "wb"), m2v_A)
        else:
            m2v_A = np.load(open(A_list_file, "rb"))
            m2v_lu = []
            for i in range(0, len(m2v_A)):
                m2v_lu.append(scipy.sparse.linalg.splu(
                    m2v_A[i].transpose().dot(m2v_A[i]).tocsc()))
        print(' [**] finish loading m2d2v matrix in %fs.' %
              (time.time() - start))
        return [m2v_A, m2v_lu]

    # ----------------------------------------------------------------------------------------
    '''pre compute for Hybrid: all deform + vertex'''
    # ----------------------------------------------------------------------------------------

    def all_dav(self):
        print(' [**] begin load all_dav mapping matrix')
        start = time.time()
        A_list_file = self.NPYpath + \
            "all_dav_A_list%d.npy" % self.paras['mapping_version']
        if self.paras['reload_all_dav']:
            m2v_A = []
            m2v_lu = []
            # for each measures
            for i in range(0, self.measure_num):
                print('  processing %d measure...' % i)
                other_vertex = [c for c in range(
                    0, self.vertex_num) if c not in self.m2v[i][2]]
                # for vertex of pure faces
                off = self.vertex_num * 3
                shape = (len(self.m2f[i][0]) * 9 + len(other_vertex)
                         * 3, (self.vertex_num + len(self.m2f[i][0])) * 3)
                # calculate A and lu for each measures
                data = []
                row = []
                col = []
                r = 0
                # for deform of pure faces
                for j in range(0, len(self.m2f[i][0])):
                    coeff = self.construct_coeff_mat(
                        self.d_inv_mean[self.m2f[i][0][j]])
                    f = [c - 1 for c in self.o_faces[3 * self.m2f[i]
                                                     [0][j]:3 * self.m2f[i][0][j] + 3, 0]]
                    v1 = range(f[0] * 3, f[0] * 3 + 3)
                    v2 = range(f[1] * 3, f[1] * 3 + 3)
                    v3 = range(f[2] * 3, f[2] * 3 + 3)
                    v4 = range(off + j * 3, off + j * 3 + 3)
                    for k in range(0, 3):
                        data += [c for c in coeff.flat]
                        row += [r, r, r, r, r + 1, r + 1, r + 1,
                                r + 1, r + 2, r + 2, r + 2, r + 2]
                        col += [v1[k], v2[k], v3[k], v4[k], v1[k], v2[k],
                                v3[k], v4[k], v1[k], v2[k], v3[k], v4[k]]
                        r += 3
                # for vertex of other vertex
                for j in range(0, len(other_vertex)):
                    v = range(other_vertex[j] * 3, other_vertex[j] * 3 + 3)
                    data += [1, 1, 1]
                    row += [r, r + 1, r + 2]
                    col += [v[0], v[1], v[2]]
                    r += 3
                A = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
                lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
                m2v_A.append(A)
                m2v_lu.append(lu)
            np.save(open(A_list_file, "wb"), m2v_A)
        else:
            m2v_A = np.load(open(A_list_file, "rb"))
            m2v_lu = []
            for i in range(0, len(m2v_A)):
                m2v_lu.append(scipy.sparse.linalg.splu(
                    m2v_A[i].transpose().dot(m2v_A[i]).tocsc()))
        print(' [**] finish loading m2d2v matrix in %fs.' %
              (time.time() - start))
        return [m2v_A, m2v_lu]

    # ----------------------------------------------------------------------------------------
    '''pre compute for Hybrid: part deform + vertex'''
    # ----------------------------------------------------------------------------------------

    def part_dav(self):
        print(' [**] begin load part_dav mapping matrix')
        start = time.time()
        A_list_file = self.NPYpath + \
            "part_dav_A_list%d.npy" % self.paras['mapping_version']
        if self.paras['reload_part_dav']:
            m2v_A = []
            m2v_lu = []
            # for each measures
            for i in range(0, self.measure_num):
                print('  processing %d measure...' % i)
                vertex_set = self.m2v[i][0]
                edge_set = self.m2v[i][1]
                # for vertex of pure faces
                off = len(vertex_set) * 3
                shape = (len(self.m2f[i][0]) * 9 + len(vertex_set)
                         * 3, (len(vertex_set) + len(self.m2f[i][0])) * 3)
                # calculate A and lu for each measures
                data = []
                row = []
                col = []
                r = 0
                # for deform of pure faces
                for j in range(0, len(self.m2f[i][0])):
                    coeff = self.construct_coeff_mat(
                        self.d_inv_mean[self.m2f[i][0][j]])
                    f = [c - 1 for c in self.o_faces[3 * self.m2f[i]
                                                     [0][j]:3 * self.m2f[i][0][j] + 3, 0]]
                    v1 = range(f[0] * 3, f[0] * 3 + 3)
                    v2 = range(f[1] * 3, f[1] * 3 + 3)
                    v3 = range(f[2] * 3, f[2] * 3 + 3)
                    v4 = range(off + j * 3, off + j * 3 + 3)
                    for k in range(0, 3):
                        data += [c for c in coeff.flat]
                        row += [r, r, r, r, r + 1, r + 1, r + 1,
                                r + 1, r + 2, r + 2, r + 2, r + 2]
                        col += [v1[k], v2[k], v3[k], v4[k], v1[k], v2[k],
                                v3[k], v4[k], v1[k], v2[k], v3[k], v4[k]]
                        r += 3
                # for vertex of ther vertex
                for j in range(0, len(edge_set)):
                    v = range(vertex_set.index(
                        edge_set[j]) * 3, vertex_set.index(edge_set[j]) * 3 + 3)
                    data += [1, 1, 1]
                    row += [r, r + 1, r + 2]
                    col += [v[0], v[1], v[2]]
                    r += 3
                A = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
                lu = scipy.sparse.linalg.splu(A.transpose().dot(A).tocsc())
                m2v_A.append(A)
                m2v_lu.append(lu)
            np.save(open(A_list_file, "wb"), m2v_A)
        else:
            m2v_A = np.load(open(A_list_file, "rb"))
            m2v_lu = []
            for i in range(0, len(m2v_A)):
                m2v_lu.append(scipy.sparse.linalg.splu(
                    m2v_A[i].transpose().dot(m2v_A[i]).tocsc()))
        print(' [**] finish loading m2d2v matrix in %fs.' %
              (time.time() - start))
        return [m2v_A, m2v_lu]


#######################################################################
#######################################################################
if __name__ == "__main__":
    filename = "../parameter.json"
    data = rawData(filename)
    data = basisData(data)
    masker = Masker(data)
    model = dataModel(data)
