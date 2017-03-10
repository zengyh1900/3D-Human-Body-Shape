#!/usr/bin/python
# coding=utf-8

import scipy.sparse.linalg
import scipy.sparse
import scipy
import numpy as np
import np.matlib
import time
import json
import os


# define the singleton pattern for dataModel
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# A rawData contains all original data from dataset:
#     o_normals, o_file_list, o_faces, vertex(o, t, mean, std)
#     measure(o, t, mean, std), control point
class MetaData:
    __metaclass__ = Singleton

    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.paras = json.load(f)
        self.data_path = self.paras['dataPath'] + "meta/"
        self.female_obj = self.paras['dataPath'] + "female/"
        self.male_obj = self.paras['dataPath'] + "male/"

        self.measure_str = self.paras['measure_str']
        self.measure_num = self.paras['measure_num']
        self.vertex_num = self.paras['vertex_num']
        self.face_num = self.paras['face_num']

        # load all data
        self.o_normals = np.load(open(self.data_path + "Normals.npy", 'rb'))

        [self.o_file_list, self.o_faces, self.o_vertex,
         self.t_vertex, self.mean_vertex, self.std_vertex] =\
            self.load_vertex_data()
        [self.o_measures, self.t_measures, self.mean_measures,
         self.std_measures] = \
            self.load_measure_data()
        self.control_point = self.load_control_point()

    # ---------------------------------------------------------------
    '''loading data: o_file_list, o_faces, vertex(o, t, mean, std)'''
    # ---------------------------------------------------------------

    def load_vertex_data(self, file_path):
        print(' [**] begin load_vertex_data ... ')
        start = time.time()
        file_list_file = self.rawDataPath + 'o_file_list.npy'
        o_face_file = self.rawDataPath + 'o_faces.npy'
        o_vertex_file = self.rawDataPath + 'o_vertex.npy'
        t_vertex_file = self.rawDataPath + 't_vertex.npy'
        mean_vertex_file = self.rawDataPath + 'mean_vertex.npy'
        std_vertex_file = self.rawDataPath + 'std_vertex.npy'
        if self.paras['reload_vertex_data']:
            str = open(self.rawDataPath + 'template.txt', 'r').read()
            o_faces = np.array(
                list(map(int, str.replace('\n', ' ').replace('f', '').split())))
            print(o_faces.shape)
            o_faces.shape = (self.face_num * 3, 1)
            o_file_list = os.listdir(self.dataSetPath)
            # load original data
            o_vertex = np.matlib.zeros((self.vertex_num * 3, len(o_file_list)))
            for i, obj in enumerate(o_file_list):
                print("load vertex from file : NO. ", i)
                tmp = open(self.dataSetPath + obj, 'r').read()
                tmp = tmp[tmp.index('\nv'): tmp.index(
                    "\nf") - 1].replace('v', '')
                tmp = list(map(float, tmp.replace('\n', ' ').split()))
                v = np.array(tmp).reshape(self.vertex_num * 3, 1)
                #-------------------------------------------------
                v.shape = (self.vertex_num, 3)
                v_mean = np.mean(v, axis=0)
                for j in range(0, self.vertex_num):
                    v[j, :] -= v_mean
                v.shape = (self.vertex_num * 3, 1)
                #-------------------------------------------------
                o_vertex[:, i] = v
            # normalize data
            mean_vertex = np.array(o_vertex.mean(axis=1))
            std_vertex = np.std(o_vertex, axis=1).reshape(o_vertex.shape[0], 1)
            t_vertex = o_vertex.copy()
            for i in range(0, o_vertex.shape[1]):
                t_vertex[:, i] -= mean_vertex
            for i in range(0, t_vertex.shape[0]):
                t_vertex[i, :] /= std_vertex[i, 0]
            # save array
            mean_vertex.shape = (self.vertex_num * 3, 1)
            self.save_obj(self.rawDataPath + 'mean.obj', mean_vertex, o_faces)
            self.save_NPY([file_list_file, o_face_file, o_vertex_file, t_vertex_file, mean_vertex_file, std_vertex_file],
                          [o_file_list, o_faces, o_vertex, t_vertex, mean_vertex, std_vertex])
            print(' [**] finish load_vertex_data from %s in %fs' %
                  (self.rawDataPath, time.time() - start))
            return [o_file_list, o_faces, o_vertex, t_vertex, mean_vertex, std_vertex]
        else:
            print(' [**] finish load_vertex_data from %s in %fs' %
                  (self.rawDataPath, time.time() - start))
            return self.load_NPY([file_list_file, o_face_file, o_vertex_file,
                                  t_vertex_file, mean_vertex_file, std_vertex_file])

    # -------------------------------------------------------
    '''load the measure data of female data set'''
    # -------------------------------------------------------

    def load_measure_data(self):
        print(' [**] begin load_measure_data ... ')
        o_measures_file = self.rawDataPath + 'o_measures.npy'
        t_measures_file = self.rawDataPath + 't_measures.npy'
        mean_measures_file = self.rawDataPath + 'mean_measures.npy'
        std_measures_file = self.rawDataPath + 'std_measures.npy'
        start = time.time()
        # load measures data
        if True:  # self.paras['reload_measure_data']:
            value = open(self.rawDataPath + 'body_measures.txt',
                         'r').read().split()
            o_measures = []
            num = len(self.measure_str)
            for i in range(0, len(value), num):
                o_measures.append(list(map(float, value[i:i + num])))
            o_measures = np.array(o_measures).transpose()
            mean_measures = np.array(o_measures.mean(axis=1))
            std_measures = np.std(o_measures, axis=1)
            t_measures = o_measures.copy()
            for i in range(0, o_measures.shape[1]):
                t_measures[:, i] -= mean_measures
            for i in range(0, o_measures.shape[0]):
                t_measures[i, :] /= std_measures[i]
            mean_measures.shape = (mean_measures.size, 1)
            std_measures.shape = (std_measures.size, 1)
            self.save_NPY([o_measures_file, t_measures_file, mean_measures_file, std_measures_file],
                          [o_measures, t_measures, mean_measures, std_measures])
            print(' [**] finish load_measure_data from %s in %fs' %
                  (self.rawDataPath, time.time() - start))
            return [o_measures, t_measures, mean_measures, std_measures]
        else:
            print(' [**] finish load_measure_data from %s in %fs' %
                  (self.rawDataPath, time.time() - start))
            return self.load_NPY([o_measures_file, t_measures_file, mean_measures_file, std_measures_file])

    # ----------------------------------------------------------------------------
    '''loading control point on the body for calculate measure from body shape'''
    # ----------------------------------------------------------------------------

    def load_control_point(self):
        print(' [**] begin load_control_point ... ')
        o_control_point_file = self.rawDataPath + 'o_control_point.npy'
        start = time.time()
        if self.paras['reload_cp']:
            f = open(self.rawDataPath + 'body_control_points.txt')
            tmplist = []
            control_point = []
            for line in f.readlines():
                if line[0] == '#':
                    if len(tmplist) != 0:
                        control_point.append(tmplist)
                        tmplist = []
                elif len(line.split()) == 1:
                    continue
                else:
                    tmplist.append(list(map(float, line.strip().split())))
            control_point.append(tmplist)
            self.save_NPY([o_control_point_file], [control_point])
            print(' [**] finish load_control_point from %s in %fs' %
                  (self.rawDataPath, time.time() - start))
            return control_point
        else:
            print(' [**] finish load_control_point from %s in %fs' %
                  (self.rawDataPath, time.time() - start))
            return self.load_NPY([o_control_point_file])[0]

    # --------------------------------------------------------------
    '''calculate measure data from given vertex by control points'''
    # --------------------------------------------------------------

    def calc_measures(self, vertex):
        measure_list = []
        # clac weight
        vol = 0.0
        kHumanbodyIntensity = 1026.0
        for i in range(0, self.face_num):
            f = [c - 1 for c in self.o_faces[3 * i:3 * i + 3, 0]]
            v0 = vertex[3 * f[0]:3 * f[0] + 3, 0]
            v1 = vertex[3 * f[1]:3 * f[1] + 3, 0]
            v2 = vertex[3 * f[2]:3 * f[2] + 3, 0]
            vol += np.cross(v0, v1).dot(v2)
        vol = abs(vol) / 6.0
        weight = kHumanbodyIntensity * vol
        weight = weight**(1.0 / 3.0) * 1000
        measure_list.append(weight)
        # calc other measures
        for measure in self.control_point:
            length = 0.0
            p2 = vertex[3 * int(measure[0][1]): 3 * int(measure[0][1]) + 3]
            for i in range(1, len(measure)):
                p1 = p2
                if measure[i][0] == 1:
                    p2 = vertex[3 * int(measure[i][1]): 3 *
                                int(measure[i][1]) + 3]
                elif measure[i][0] == 2:
                    p2 = vertex[3 * int(measure[i][1]) : 3 * int(measure[i][1]) + 3] *\
                        measure[i][3] + \
                        vertex[3 * int(measure[i][2]) : 3 * int(measure[i][2]) + 3] *\
                        measure[i][4]
                else:
                    p2 = vertex[3 * int(measure[i][1]) : 3 * int(measure[i][1]) + 3] *\
                        measure[i][4] + \
                        vertex[3 * int(measure[i][2]) : 3 * int(measure[i][2]) + 3] *\
                        measure[i][5] + \
                        vertex[3 * int(measure[i][3]) : 3 * int(measure[i][3]) + 3] *\
                        measure[i][6]
                length += np.sqrt(np.sum((p1 - p2)**2.0))
            measure_list.append(length * 1000)
        return np.array(measure_list).reshape(len(measure_list), 1)

    # ----------------------------------------------------------------
    '''build sparse matrix'''
    # ----------------------------------------------------------------

    def build_equation(self, m_datas, basis_num):
        shape = (m_datas.shape[1] * basis_num, m_datas.shape[0] * basis_num)
        data = []
        rowid = []
        colid = []
        for i in range(0, m_datas.shape[1]):  # 1531
            for j in range(0, basis_num):  # 10
                data += [c for c in m_datas[:, i].flat]
                rowid += [basis_num * i + j for a in range(m_datas.shape[0])]
                colid += [a for a in range(j * m_datas.shape[0],
                                           (j + 1) * m_datas.shape[0])]
        # print (' [**] data: ', len(data), ', rowid: ', len(rowid), ', colid: ', len(colid))
        return scipy.sparse.coo_matrix((data, (rowid, colid)), shape)

    # --------------------------------------------------------------
    '''save obj file'''
    # --------------------------------------------------------------

    def save_obj(self, filename, v, f):
        file = open(filename, 'w')
        file.write('# %d vertices and %d faces\n' %
                   (v.shape[0] / 3, f.shape[0] / 3))
        for i in range(0, v.shape[0], 3):
            file.write('v %f %f %f\n' % (v[i][0], v[i + 1][0], v[i + 2][0]))
        for i in range(0, f.shape[0], 3):
            file.write('f %d %d %d\n' % (f[i][0], f[i + 1][0], f[i + 2][0]))
        tmp = v[2::3, 0]
        print('      save ' + filename + "  ,height: ", tmp.max() - tmp.min())
        file.close()

    # --------------------------------------------------------------
    '''save NPY file'''
    # --------------------------------------------------------------

    def save_NPY(self, name_list, obj_list):
        for i in range(0, len(name_list)):
            print('  saving ', name_list[i], ' ing...')
            np.save(open(name_list[i], 'wb'), obj_list[i])

    # --------------------------------------------------------------
    '''load NPY file'''
    # --------------------------------------------------------------

    def load_NPY(self, name_list):
        obj_list = []
        for i in range(0, len(name_list)):
            print('  loading ', name_list[i], ' ing...')
            obj = np.load(open(name_list[i], 'rb'))
            obj_list.append(obj)
        return obj_list

    # -------------------------------------------------------
    '''calculating normals '''
    # -------------------------------------------------------

    def ComputeNormals(self):
        self.normals = []
        vertexNormalLists = [[] for i in range(0, len(self.o_vertex))]
        for face in self.o_faces:
            AB = np.array(self.o_vertex[face[0]]) - \
                np.array(self.o_vertex[face[1]])
            AC = np.array(self.o_vertex[face[0]]) - \
                np.array(self.o_vertex[face[2]])
            n = np.cross(AB, -AC)
            n /= np.linalg.norm(n)
            for i in range(0, 3):
                vertexNormalLists[face[i]].append(n)
        for idx, normalList in enumerate(vertexNormalLists):
            normalSum = np.zeros(3)
            for normal in normalList:
                normalSum += normal
            normal = normalSum / float(len(normalList))
            normal /= np.linalg.norm(normal)
            self.normals.append(map(float, normal.tolist()))


#######################################################################
#######################################################################
if __name__ == "__main__":
    filename = "../parameter.json"
    data = MetaData(filename)
    data.save_obj("../../result/mean.obj", data.mean_vertex, data.o_faces)

    tmp = data.mean_vertex.copy()
    tmp.shape = (data.vertex_num, 3)
    print(np.mean(tmp, axis=0))
