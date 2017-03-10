#!/usr/bin/python
#coding=utf-8

import sys
sys.path.append("..")
from dataProcess.dataModel import *
import numpy as np


class deformHybrid:
    '''
        a model map the changed measures to correlated faces
        and keep others unchanged
    '''
    __metaclass__ = Singleton
    def __init__(self, data):
        self.type = "deform-hybrid"
        self.data = data
        self.prev = []
        self.deformation = None
        self.vertex = None
        self.switcher = {
                1: self.all_dad,
                2: self.v_part_dad,
                3: self.all_dav,
                4: self.part_dav,
        }
        self.demo_num = self.data.measure_num

    # ------------------------------------------------------------------------------------
    '''given measures(normalized), mapping to deformation, using hybrid method '''
    # ------------------------------------------------------------------------------------
    def mapping(self, data):
        diff = []
        for i in range(0, len(self.prev)):
            if self.prev[i] != data[i]:
                diff.append(i)
        self.prev = [c for c in data]
        measures = np.array(data).reshape(self.data.measure_num, 1)
        measures = self.data.mean_measures + self.data.std_measures * measures

        if len(diff) != 1:
            print ("deform-local pure synthesize")
            deform = []
            for i in range(0, self.data.face_num):
                mask = np.array(self.data.mask[i * 19:i * 19 + 19, 0]).reshape(19, 1)
                data = np.array(measures[mask])
                data.shape = (data.size, 1)
                s = self.data.L_list[i].dot(data)
                deform += [a for a in s.flat]
            deform = np.array(deform).reshape(len(deform), 1)
            self.deformation = deform
            [self.vertex, n, f] = self.data.d_synthesize(self.deformation)
            return [self.vertex, n, f]
        else:
            return self.switcher.get(self.data.paras['Hybrid_method'], self.all_dad)(diff[0], measures)

    # ------------------------------------------------------------------------------------
    '''given measures(normalized), mapping to deformation, using hybrid method '''
    # ------------------------------------------------------------------------------------
    def all_dad(self, index, measures):
        print ("  deform-local hybrid method: all deform + deform: ")
        sum_list = self.data.m2f[index][0] + self.data.m2f[index][1]
        print ("  %s, correlated faces num: %d"%(self.data.measure_str[index] , len(sum_list)))
        for i in range(0, len(sum_list)):
            k = sum_list[i]  # No. of face
            mask = np.array(self.data.mask[k * 19:k * 19 + 19, 0]).reshape(19, 1)
            data = np.array(measures[mask])
            data.shape = (data.size, 1)
            s = self.data.L_list[k].dot(data)
            self.deformation[9*k:9*k+9,0] = [c for c in s.flat]
        [self.vertex, n, f] =  self.data.d_synthesize(self.deformation)
        return [self.vertex, n, f]


    # ------------------------------------------------------------------------------------
    '''part deform + deform , using measure->part->face_>vertex'''
    # ------------------------------------------------------------------------------------
    def v_part_dad(self, index, measures):
        pure_face = self.data.face_vertex[index][0]
        edge_face = self.data.face_vertex[index][1]
        vertex_set = self.data.face_vertex[index][2]
        print ("  %s, pure faces num: %d, edge faces num: %d" % \
              (self.data.measure_str[index], len(pure_face), len(edge_face)))
        deform = []
        # for pure faces
        for i in range(0, len(pure_face)):
            k = pure_face[i]
            mask = np.array(self.data.mask[k * 19:k * 19 + 19, 0]).reshape(19, 1)
            data = np.array(measures[mask])
            data.shape = (data.size, 1)
            s = self.data.L_list[k].dot(data)
            deform += [a for a in s.flat]
            self.deformation[9 * k:9 * k + 9, 0] = [a for a in s.flat]
        # for edge faces
        for i in range(0, len(edge_face)):
            k = edge_face[i]
            s = self.deformation[k*9:k*9+9, 0]
            deform += [a for a in s.flat]
        # merge new vertex to old vertex
        deform = np.array(deform).reshape(len(deform), 1)
        Atd = self.data.m2v_A[index].transpose().dot(deform)
        mat = self.data.m2v_lu[index].solve(Atd)
        # ===================debug===========================
        # rebuild = (self.data.m2v_A[index].transpose().dot(self.data.m2v_A[index])).dot(mat)
        # error = Atd - rebuild
        # print self.vertex[:100,0]
        # ==================================================
        mat = np.array(mat[:len(vertex_set) * 3, 0]).reshape(len(vertex_set) * 3, 1)
        for i in range(0, len(vertex_set)):
            k = vertex_set[i]
            self.vertex[k * 3:k * 3 + 3, 0] = mat[i * 3:i * 3 + 3, 0]
        return [self.vertex, -self.data.o_normals, self.data.o_faces - 1]

    # ------------------------------------------------------------------------------------
    '''given measures(normalized), mapping to deformation, using hybrid method '''
    # ------------------------------------------------------------------------------------
    def part_dad(self, index, measures):
        print ("  deform-local hybrid method: part deform + deform: ")
        l1 = len(self.data.m2f[index][0])
        l2 = len(self.data.m2f[index][1])
        print ("  %s, pure faces num: %d, edge faces num: %d" % (self.data.measure_str[index], l1, l2))
        deform = []
        # for pure faces
        for i in range(0, l1):
            k = self.data.m2f[index][0][i]
            mask = np.array(self.data.mask[k * 19:k * 19 + 19, 0]).reshape(19, 1)
            data = np.array(measures[mask])
            data.shape = (data.size, 1)
            s = self.data.L_list[k].dot(data)
            deform += [a for a in s.flat]
            self.deformation[9 * k:9 * k + 9, 0] = [a for a in s.flat]
        # for edge faces
        for i in range(0, l2):
            k = self.data.m2f[index][1][i]
            s = self.deformation[k*9:k*9+9, 0]
            deform += [a for a in s.flat]
        # merge new vertex to old vertex
        vertex_set = self.data.m2v_list[index][0]
        deform = np.array(deform).reshape(len(deform), 1)
        Atd = self.data.m2v_A[index].transpose().dot(deform)
        mat = self.data.m2v_lu[index].solve(Atd)
        # ===================debug===========================
        rebuild = (self.data.m2v_A[index].transpose().dot(self.data.m2v_A[index])).dot(mat)
        error = Atd - rebuild
        print (self.vertex[:100,0])
        # ==================================================
        mat = np.array(mat[:len(vertex_set) * 3, 0]).reshape(len(vertex_set) * 3, 1)
        for i in range(0, len(vertex_set)):
            k = vertex_set[i]
            self.vertex[k * 3:k * 3 + 3, 0] = mat[i * 3:i * 3 + 3, 0]
        return [self.vertex, -self.data.o_normals, self.data.o_faces - 1]

    # ------------------------------------------------------------------------------------
    '''given measures(normalized), mapping to deformation, using hybrid method '''
    # ------------------------------------------------------------------------------------
    def all_dav(self, index, measures):
        print ("  deform-local hybrid method: all deform + vertex: ")
        other_vertex = [c for c in range(0, self.data.vertex_num) if c not in self.data.m2v_list[index][2]]
        print ("  %s, pure faces num: %d, other vertex num: %d" % \
              (self.data.measure_str[index], len(self.data.m2f[index][0]), len(other_vertex)))
        deform = []
        # for deformation of pure faces
        for i in range(0, len(self.data.m2f[index][0])):
            k = self.data.m2f[index][0][i]
            mask = np.array(self.data.mask[k * 19:k * 19 + 19, 0]).reshape(19, 1)
            data = np.array(measures[mask])
            data.shape = (data.size, 1)
            s = self.data.L_list[k].dot(data)
            deform += [a for a in s.flat]
            self.deformation[9 * k:9 * k + 9, 0] = [a for a in s.flat]
        # for other vertex
        for i in range(0, len(other_vertex)):
            k = other_vertex[i]
            s = self.vertex[3 * k:3 * k + 3, 0]
            deform += [a for a in s.flat]
        deform = np.array(deform).reshape(len(deform), 1)
        Atd = self.data.m2v_A[index].transpose().dot(deform)
        mat = self.data.m2v_lu[index].solve(Atd)
        mat = np.array(mat[:self.data.vertex_num * 3, :]).reshape(self.data.vertex_num * 3, 1)
        # merge
        self.vertex = mat
        return [self.vertex, -self.data.o_normals, self.data.o_faces - 1]

    # ------------------------------------------------------------------------------------
    '''given measures(normalized), mapping to deformation, using hybrid method '''
    # ------------------------------------------------------------------------------------
    def part_dav(self, index, measures):
        print ("  deform-local hybrid method: part deform + vertex: ")
        edge_set = self.data.m2v_list[index][1]
        vertex_set = self.data.m2v_list[index][0]
        print ("  %s, pure faces num: %d, edge vertex num: %d" % \
              (self.data.measure_str[index], len(self.data.m2f[index][0]), len(edge_set)))
        deform = []
        # for deformation of pure faces
        for i in range(0, len(self.data.m2f[index][0])):
            k = self.data.m2f[index][0][i]
            mask = np.array(self.data.mask[k * 19:k * 19 + 19, 0]).reshape(19, 1)
            data = np.array(measures[mask])
            data.shape = (data.size, 1)
            s = self.data.L_list[k].dot(data)
            deform += [a for a in s.flat]
            self.deformation[9 * k:9 * k + 9, 0] = [a for a in s.flat]
        # for other vertex
        for i in range(0, len(edge_set)):
            k = edge_set[i]
            s = self.vertex[3 * k:3 * k + 3, 0]
            deform += [a for a in s.flat]
        deform = np.array(deform).reshape(len(deform), 1)
        Atd = self.data.m2v_A[index].transpose().dot(deform)
        mat = self.data.m2v_lu[index].solve(Atd)
        mat = np.array(mat[:len(vertex_set) * 3, :]).reshape(len(vertex_set) * 3, 1)
        # merge
        for i in range(0, len(vertex_set)):
            k = vertex_set[i]
            self.vertex[k * 3:k * 3 + 3, 0] = mat[i * 3:i * 3 + 3, 0]
        return [self.vertex, -self.data.o_normals, self.data.o_faces - 1]


#############################################
'''test'''
#############################################
if __name__ == "__main__":
    filename = "../parameter.json"
    data = rawData(filename)
    bd = basisData(data)
    mark = Masker(data)
    measurePredict = measureMining(data)
    dm = dataModel(bd, mark, measurePredict)

    dl = deformHybrid(dm)
