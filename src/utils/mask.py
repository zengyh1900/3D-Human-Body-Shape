#!/usr/bin/python
# coding=utf-8

import sys
sys.path.append("..")
import numpy.matlib
from dataProcess.rawData import *


class Masker:
    '''
        A Masker contains data of local mapping
        color_list, color_set, p2m, p2p, p2f, m2p, m2f, m2v, mask
    '''
    __metaclass__ = Singleton

    def __init__(self, data):
        self.data = data
        self.dataPath = self.data.paras['dataPath'] + "NPYdata/localMapper/"
        self.mapVersion = self.data.paras['mapping_version']
        self.part_num = self.data.paras["part_num"]

        [self.color_list, self.color_set] = self.getColor()
        [self.p2m, self.p2p, self.p2f, self.m2p,
            self.m2f, self.m2v, self.mask] = self.getMap()
        self.face_vertex = self.calc_m2p2f2v()

    # ---------------------------------------------------------------
    '''loading information of color'''
    # ---------------------------------------------------------------

    def getColor(self):
        print(' [**] begin loading color data ...')
        color_list_path = self.dataPath + "color_list.npy"
        color_set_path = self.dataPath + "color_set.npy"
        if self.data.paras['reload_color']:
            tmp = open(self.dataPath + 'body_part.obj', 'r').read()
            tmp = tmp[tmp.index('\nv'): tmp.index("\n#!") - 1].replace('v', '')
            tmp = list(map(float, tmp.replace('\n', ' ').split()))
            body_part = np.array(tmp).reshape(self.data.vertex_num, 6)
            body_part = np.array(body_part[:, 3:])
            tmp = set()
            color_list = []
            for i in range(0, self.data.vertex_num):
                tmp.add((body_part[i, 0], body_part[i, 1], body_part[i, 2]))
                color_list.append(
                    (body_part[i, 0], body_part[i, 1], body_part[i, 2]))
            color_set = [item for item in tmp]
            self.data.save_NPY([color_list_path, color_set_path], [
                               color_list, color_set])
        else:
            [list_tmp, set_tmp] = self.data.load_NPY(
                [color_list_path, color_set_path])
            color_set = []
            for i in range(0, set_tmp.shape[0]):
                color_set.append((set_tmp[i, 0], set_tmp[i, 1], set_tmp[i, 2]))
            color_list = []
            for i in range(0, list_tmp.shape[0]):
                color_list.append(
                    (list_tmp[i, 0], list_tmp[i, 1], list_tmp[i, 2]))
        print(' [**] finish loading color data.')
        return [color_list, color_set]

    # ---------------------------------------------------------------
    '''loading information of p2m, p2p, p2f, m2p, m2f, m2v'''
    # ---------------------------------------------------------------

    def getMap(self):
        print(' [**] begin loading color data ...')
        p2m_path = "part2measure_v%d" % self.mapVersion
        p2f_path = self.dataPath + "p2f.npy"
        m2f_path = self.dataPath + ("m2f%d.npy" % self.mapVersion)
        m2v_path = self.dataPath + ("m2v%d.npy" % self.mapVersion)
        mask_path = self.dataPath + ("mask%d.npy" % self.mapVersion)
        p2m = self.data.paras[p2m_path]
        p2p = self.data.paras["part2part"]
        # loading measure2part
        m2p = []
        m2f = []
        m2v = []
        p2f = []
        mask = np.matlib.zeros((19 * self.data.face_num, 1), dtype=bool)
        for i in range(0, self.data.measure_num):
            m2p.append([])
            m2f.append([[], []])
        for i in range(0, self.part_num):
            p2f.append([])
        for i in range(0, len(p2m)):
            for j in range(0, len(p2m[i])):
                m2p[p2m[i][j]].append(i)
        # load p2f, m2f and m2v
        if self.data.paras['reload_mask']:
            for i in range(0, self.data.face_num):
                print('  processing face: NO. %d' % i)
                f = [c - 1 for c in self.data.o_faces[3 * i:3 * i + 3, 0]]
                p_tmp = set()
                v3_list = []
                for j in range(0, len(f)):
                    m_tmp = set()
                    kind = self.color_set.index(self.color_list[f[j]])
                    for k in p2m[kind]:
                        m_tmp.add(k)
                    v3_list.append(m_tmp)
                    p_tmp.add(kind)
                for j in p_tmp:
                    p2f[j].append(i)
                all_measure = v3_list[0] | v3_list[1] | v3_list[2]
                pure_measure = v3_list[0] & v3_list[1] & v3_list[2]
                edge_measure = list(all_measure - pure_measure)
                pure_measure = list(pure_measure)
                for j in pure_measure:
                    m2f[j][0].append(i)
                for j in edge_measure:
                    m2f[j][1].append(i)
                for j in all_measure:
                    mask[19 * i + j, 0] = 1
            # load m2v
            for i in range(0, self.data.measure_num):
                print('  processing measure: NO. %d' % i)
                # for vertex of pure faces
                pure_vertex_set = set()
                for j in range(0, len(m2f[i][0])):
                    k = m2f[i][0][j]  # No. of face
                    pure_vertex_set.add(self.data.o_faces[3 * k, 0] - 1)
                    pure_vertex_set.add(self.data.o_faces[3 * k + 1, 0] - 1)
                    pure_vertex_set.add(self.data.o_faces[3 * k + 2, 0] - 1)
                # for vertex of edge faces
                edge_vertex_set = set()
                for j in range(0, len(m2f[i][1])):
                    k = m2f[i][1][j]  # No. of face
                    edge_vertex_set.add(self.data.o_faces[3 * k, 0] - 1)
                    edge_vertex_set.add(self.data.o_faces[3 * k + 1, 0] - 1)
                    edge_vertex_set.add(self.data.o_faces[3 * k + 2, 0] - 1)
                vertex_set = pure_vertex_set | edge_vertex_set
                edge_vertex_set = list(vertex_set - pure_vertex_set)
                pure_vertex_set = list(pure_vertex_set)
                vertex_set = list(vertex_set)
                m2v.append([vertex_set, edge_vertex_set, pure_vertex_set])
            self.data.save_NPY([p2f_path, m2f_path, m2v_path, mask_path], [
                               p2f, m2f, m2v, mask])
        else:
            [p2f, m2f, m2v, mask] = self.data.load_NPY(
                [p2f_path, m2f_path, m2v_path, mask_path])
        print(' [**] finish loading color data.')
        return [p2m, p2p, p2f, m2p, m2f, m2v, mask]

    # ----------------------------------------------------------------------------------------
    '''pre calculate measure->part->face->vertex_set'''
    # ----------------------------------------------------------------------------------------

    def calc_m2p2f2v(self):
        face_vertex_file = self.dataPath + "m2p2f2v%d" % (self.mapVersion)
        print(' [**] begin load m2p2f2v....')
        start = time.time()
        if self.data.paras["reload_m2p2f2v"]:
            face_vertex = []
            # for each measures
            for i in range(0, self.data.measure_num):
                print('  processing %d measure...' % i)
                part = self.m2p[i]
                edge_part = set()
                pure_face = set()
                edge_face = set()
                vertex_set = set()
                for j in part:
                    for k in self.p2p[j]:
                        edge_part.add(k)
                    for k in range(0, len(self.p2f[j])):
                        pure_face.add(self.p2f[j][k])
                        vertex_set.add(self.data.o_faces[
                                       self.p2f[j][k] * 3, 0] - 1)
                        vertex_set.add(self.data.o_faces[
                                       self.p2f[j][k] * 3 + 1, 0] - 1)
                        vertex_set.add(self.data.o_faces[
                                       self.p2f[j][k] * 3 + 2, 0] - 1)
                edge_part = [c for c in edge_part if c not in part]
                for j in edge_part:
                    for k in range(0, len(self.p2f[j])):
                        edge_face.add(self.p2f[j][k])
                        vertex_set.add(self.data.o_faces[
                                       self.p2f[j][k] * 3, 0] - 1)
                        vertex_set.add(self.data.o_faces[
                                       self.p2f[j][k] * 3 + 1, 0] - 1)
                        vertex_set.add(self.data.o_faces[
                                       self.p2f[j][k] * 3 + 2, 0] - 1)
                vertex_set = list(vertex_set)
                pure_face = list(pure_face)
                edge_face = list(edge_face)
                # edge_face = [c for c in range(0, 25000) if c not in pure_face]
                # vertex_set = range(0, 12500)
                face_vertex.append([pure_face, edge_face, vertex_set])
            self.data.save_NPY([face_vertex_file], [face_vertex])
        else:
            face_vertex = self.data.load_NPY([face_vertex_file])[0]
        print(' [**] finish loading m2p2f2v in %fs' % (time.time() - start))
        return face_vertex

#######################################################################
#######################################################################
if __name__ == "__main__":
    filename = "../parameter.json"
    data = rawData(filename)
    mask = Masker(data)
    print(mask.color_set)
