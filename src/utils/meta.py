#!/usr/bin/python
# coding=utf-8

import scipy.sparse.linalg
import scipy.sparse
import scipy
import numpy.matlib
import numpy
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


# A MetaData contains all original data from dataset:
#     normals, file_list, faces, vertex(o, t, mean, std)
#     measure(o, t, mean, std), control point
class MetaData:
    __metaclass__ = Singleton

    def __init__(self, filename, flag):
        with open(filename, 'r') as f:
            self.paras = json.load(f)
        self.flag_ = flag
        self.data_path = self.paras['data_path'] + "meta/"
        self.ans_path = self.paras['ans_path']

        self.measure_str = self.paras['measure_str']
        self.m_num = self.paras['m_num']
        self.v_num = self.paras['v_num']
        self.f_num = self.paras['f_num']

        # load all data
        self.cp = self.load_cp()
        [self.facet, self.normals] = self.load_template()
        [self.file_list, self.vertex, self.mean_vertex,
            self.std_vertex] = self.obj2data()
        [self.measure, self.mean_measure, self.std_measure,
            self.t_measure] = self.load_measure()
        self.body_num = len(self.file_list)

    # load normal data and facet information for female and male
    def load_template(self):
        normals = numpy.load(open(self.data_path + 'Normals.npy', 'rb'))
        facet = numpy.zeros((self.f_num, 3))
        f = open(self.data_path + 'template.txt', 'r')
        i = 0
        for line in f:
            if line[0] == 'f':
                tmp = list(map(int, line[1:].split()))
                facet[i, :] = tmp
                i += 1
        return[facet, normals]

    # loading data: file_list, vertex, mean, std
    def obj2data(self):
        print(' [**] begin obj2data about... ')
        start = time.time()
        vpath = self.data_path + "vertex_0%d.npy" % self.flag_
        fpath = self.data_path + "file_list_0%d.npy" % self.flag_
        if self.paras["reload_obj"]:
            folder = self.data_path + ("0%d/" % self.flag_)
            file_list = os.listdir(folder)
            # load original data
            vertex = numpy.zeros((len(file_list), self.v_num, 3))
            v = numpy.zeros((self.v_num, 3))
            for i, obj in enumerate(file_list):
                print("loading from folder %d in file : NO.%d" %
                      (self.flag_, i))
                f = open(folder + obj, 'r')
                j = 0
                for line in f:
                    if line[0] == '#':
                        continue
                    elif "v " in line:
                        line.replace('\n', ' ')
                        tmp = list(map(float, line[1:].split()))
                        v[j, :] = tmp
                        j += 1
                    else:
                        break
                v_mean = numpy.mean(v, axis=0)
                v -= v_mean
                vertex[i, :] = v
            numpy.save(open(vpath, "wb"), vertex)
            numpy.save(open(fpath, "wb"), file_list)
        else:
            vertex = numpy.load(open(vpath, "rb"))
            file_list = numpy.load(open(fpath, "rb"))
        # normalize data
        mean_vertex = numpy.array(vertex.mean(axis=0)).reshape(self.v_num, 3)
        std_vertex = numpy.array(vertex.std(axis=0)).reshape(self.v_num, 3)
        self.save_obj(self.ans_path + 'mean_0%d.obj' %
                      self.flag_, mean_vertex, self.facet)
        print(' [**] finish obj2data with %d in %fs.' %
              (self.flag_, time.time() - start))
        return [file_list, vertex, mean_vertex, std_vertex]

    # loading control point on the body for calculate measure from body shape'
    def load_cp(self):
        print(' [**] begin load_cp ... ')
        cp_file = self.data_path + "cp.npy"
        start = time.time()
        if self.paras['reload_cp']:
            f = open(self.data_path + 'body_control_points.txt', "r")
            tmplist = []
            cp = []
            for line in f:
                if '#' in line:
                    if len(tmplist) != 0:
                        cp.append(tmplist)
                        tmplist = []
                elif len(line.split()) == 1:
                    continue
                else:
                    tmplist.append(list(map(float, line.strip().split())))
            cp.append(tmplist)
            numpy.save(open(cp_file, "wb"), cp)
        else:
            cp = numpy.load(open(cp_file, "rb"))
        print(' [**] finish load_cp from %s in %fs' %
              (cp_file, time.time() - start))
        return cp

    # load the measure data of female data set
    def load_measure(self):
        print(' [**] begin load_measure_data ... ')
        m_file = self.data_path + ("measure_0%d.npy" % self.flag_)
        start = time.time()
        # load measures data
        if self.paras['reload_m_data']:
            measure = numpy.zeros((self.m_num, len(self.file_list)))
            for i in range(0, len(self.file_list)):
                print("  calc measure for %d of body %d" % (self.flag_, i))
                measure[:, i] = self.calc_measures(self.vertex[i, :, :]).flat
            numpy.save(open(m_file, "wb"), measure)
        else:
            measure = numpy.load(open(m_file, "rb"))
        mean_measure = numpy.array(measure.mean(axis=1)).reshape(self.m_num, 1)
        std_measure = numpy.array(measure.std(axis=1)).reshape(self.m_num, 1)
        t_measure = measure - mean_measure
        t_measure /= std_measure
        print(' [**] finish load_measure for %d in %fs' %
              (self.flag_, time.time() - start))
        return[measure, mean_measure, std_measure, t_measure]

    # calculate measure data from given vertex by control points
    def calc_measures(self, vertex):
        measure_list = []
        # clac weight
        vol = 0.0
        kHumanbodyIntensity = 1026.0
        for i in range(0, self.f_num):
            f = [c - 1 for c in self.facet[i, :]]
            v0 = vertex[f[0], :]
            v1 = vertex[f[1], :]
            v2 = vertex[f[2], :]
            vol += numpy.cross(v0, v1).dot(v2)
        vol = abs(vol) / 6.0
        weight = kHumanbodyIntensity * vol
        weight = weight**(1.0 / 3.0) * 1000
        measure_list.append(weight)
        # calc other measures
        for measure in self.cp:
            length = 0.0
            p2 = vertex[int(measure[0][1]), :]
            for i in range(1, len(measure)):
                p1 = p2
                if measure[i][0] == 1:
                    p2 = vertex[int(measure[i][1]), :]
                elif measure[i][0] == 2:
                    p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
                        vertex[int(measure[i][2]), :] * measure[i][4]
                else:
                    p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
                        vertex[int(measure[i][2]), :] * measure[i][5] + \
                        vertex[int(measure[i][3]), :] * measure[i][6]
                length += numpy.sqrt(numpy.sum((p1 - p2)**2.0))
            measure_list.append(length * 1000)
        return numpy.array(measure_list).reshape(self.m_num, 1)

    # build sparse matrix
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
        return scipy.sparse.coo_matrix((data, (rowid, colid)), shape)

    # save obj file
    def save_obj(self, filename, v, f):
        file = open(filename, 'w')
        for i in range(0, v.shape[0]):
            file.write('v')
            for j in range(0, v.shape[1]):
                file.write(" %f" % v[i][j])
            file.write("\n")
        for i in range(0, f.shape[0]):
            file.write('f %d %d %d\n' % (f[i][0], f[i][1], f[i][2]))
        tmp = v[:, 2]
        print('      save ' + filename + "  ,height: ", tmp.max() - tmp.min())
        file.close()

    # calculating normals
    def ComputeNormals(self):
        self.normals = []
        vertexNormalLists = [[] for i in range(0, len(self.o_vertex))]
        for face in self.o_faces:
            AB = numpy.array(self.o_vertex[face[0]]) - \
                numpy.array(self.o_vertex[face[1]])
            AC = numpy.array(self.o_vertex[face[0]]) - \
                numpy.array(self.o_vertex[face[2]])
            n = numpy.cross(AB, -AC)
            n /= numpy.linalg.norm(n)
            for i in range(0, 3):
                vertexNormalLists[face[i]].append(n)
        for idx, normalList in enumerate(vertexNormalLists):
            normalSum = numpy.zeros(3)
            for normal in normalList:
                normalSum += normal
            normal = normalSum / float(len(normalList))
            normal /= numpy.linalg.norm(normal)
            self.normals.append(map(float, normal.tolist()))


# test for this module
if __name__ == "__main__":
    filename = "../parameter.json"
    male_flag = 1
    female_flag = 2
    # male_body = MetaData(filename, male_flag)
    female_body = MetaData(filename, female_flag)
