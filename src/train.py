#!/usr/bin/python
# coding=utf-8

import random
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from fancyimpute import MICE
import scipy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import json
import sys
import time
import os
import utils

DATA_DIR = "../data"

# calculate normals
def compute_normals(vertex, facet):
  normals = []
  vertexNormalLists = [[] for i in range(0, len(vertex))]
  for face in facet:
    AB = np.array(vertex[face[0]]) - np.array(vertex[face[1]])
    AC = np.array(vertex[face[0]]) - np.array(vertex[face[2]])
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
    normals.append(map(float, normal.tolist()))
  return normals

# load facet information from txt file
def convert_template():
  print("starting convert_template")
  facet = np.zeros((utils.F_NUM, 3), dtype=int)
  f = open(os.path.join(DATA_DIR, 'template.txt'), 'r')
  i = 0
  for line in f:
    if line[0] == 'f':
      tmp = list(map(int, line[1:].split()))
      facet[i, :] = tmp
      i += 1
  np.save(open(os.path.join(DATA_DIR,"facet.npy"), "wb"), facet)
  print("finish convert template from txt to npy")
  return facet

# loading data: file_list, vertex, mean, std
def obj2npy(label="female"):
  print(' [**] begin obj2npy about %s... '%label)
  start = time.time()
  OBJ_DIR = os.path.join(DATA_DIR, "obj")
  obj_file_dir = os.path.join(OBJ_DIR, label)
  file_list = os.listdir(obj_file_dir)

  # load original data
  vertex = []
  for i, obj in enumerate(file_list):
    sys.stdout.write('\r>> Converting %s body %d'%(label, i))
    sys.stdout.flush()
    f = open(os.path.join(obj_file_dir, obj), 'r')
    j = 0
    for line in f:
      if line[0] == '#':
        continue
      elif "v " in line:
        line.replace('\n', ' ')
        tmp = list(map(float, line[1:].split()))
        vertex.append(tmp)
        j += 1
      else:
        break
  # normalize data
  vertex = np.array(vertex).reshape(len(file_list), utils.V_NUM, 3)
  for i in range(len(file_list)):
    v_mean = np.mean(vertex[i,:,:], axis=0)
    vertex[i,:,:] -= v_mean
  mean_vertex = np.array(vertex.mean(axis=0)).reshape(utils.V_NUM, 3)
  std_vertex = np.array(vertex.std(axis=0)).reshape(utils.V_NUM, 3)
  facet = np.load(open(os.path.join(DATA_DIR, "facet.npy"),"rb"))
  np.save(open(os.path.join(DATA_DIR, "%s_vertex.npy"%label), "wb"), vertex)
  # np.save(open(os.path.join(DATA_DIR, "%s_mean_vertex.npy"%label), "wb"), mean_vertex)
  # np.save(open(os.path.join(DATA_DIR, "%s_std_vertex.npy"%label), "wb"), std_vertex)
  print('\n [**] finish obj2npy in %fs.' %(time.time() - start))
  return [vertex, mean_vertex, std_vertex, file_list]

# convert cp from txt to npy
def convert_cp():
  print(' [**] begin load_cp ... ')
  start = time.time()
  f = open(os.path.join(DATA_DIR, 'body_control_points.txt'), "r")
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
  np.save(open(os.path.join(DATA_DIR, "cp.npy"), "wb"), cp)
  print(' [**] finish convert_cp from in %fs' %(time.time() - start))
  return cp

# calculate the measure data and convert to npy
def convert_measure(cp, vertex, facet, label="female"):
  print(' [**] begin load_measure_data ... ')
  start = time.time()
  # load measures data
  OBJ_DIR = os.path.join(DATA_DIR, "obj")
  measure = np.zeros((utils.M_NUM, vertex.shape[0]))
  for i in range(vertex.shape[0]):
    sys.stdout.write('\r>> calc %s measure of body %d'%(label, i))
    sys.stdout.flush()
    measure[:, i] = calc_measure(cp, vertex[i, :, :], facet).flat
  np.save(open(os.path.join(DATA_DIR, "%s_measure.npy"%label), "wb"), measure)
  mean_measure = np.array(measure.mean(axis=1)).reshape(utils.M_NUM, 1)
  std_measure = np.array(measure.std(axis=1)).reshape(utils.M_NUM, 1)
  t_measure = measure - mean_measure
  t_measure /= std_measure
  np.save(open(os.path.join(DATA_DIR, "%s_measure.npy"%label), "wb"), measure)
  np.save(open(os.path.join(DATA_DIR, "%s_mean_measure.npy"%label), "wb"), mean_measure)
  np.save(open(os.path.join(DATA_DIR, "%s_std_measure.npy"%label), "wb"), std_measure)
  np.save(open(os.path.join(DATA_DIR, "%s_t_measure.npy"%label), "wb"), t_measure)
  print(' [**] finish load_measure for %s in %fs' %(label, time.time() - start))
  return [measure, mean_measure, std_measure, t_measure]

# calculate measure data from given vertex by control points
def calc_measure(cp, vertex, facet):
  measure_list = []
  # clac weight
  vol = 0.0
  kHumanbodyIntensity = 1026.0
  for i in range(0, utils.F_NUM):
    f = [c - 1 for c in facet[i, :]]
    v0 = vertex[f[0], :]
    v1 = vertex[f[1], :]
    v2 = vertex[f[2], :]
    vol += np.cross(v0, v1).dot(v2)
  vol = abs(vol) / 6.0
  weight = kHumanbodyIntensity * vol
  weight = weight**(1.0 / 3.0) * 1000
  measure_list.append(weight)
  # calc other measures
  for measure in cp:
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
      length += np.sqrt(np.sum((p1 - p2)**2.0))
    measure_list.append(length * 1000)
  return np.array(measure_list).reshape(utils.M_NUM, 1)

# loading deform-based data
def load_d_data(vertex, facet, label="female"):
  print(" [**] begin load_d_data ...")
  start = time.time()
  dets = []
  mean_vertex = np.array(vertex.mean(axis=0)).reshape(utils.V_NUM, 3)
  d_inv_mean = get_inv_mean(mean_vertex, facet)
  deform = np.zeros((vertex.shape[0], utils.F_NUM, 9))
  # calculate deformation mat of each body shape
  for i in range(0, utils.F_NUM):
    sys.stdout.write('\r>> loading %s deformation of facet %d'%(label, i))
    sys.stdout.flush()
    v = [k - 1 for k in facet[i, :]]
    for j in range(0, vertex.shape[0]):
      v1 = vertex[j, v[0], :]
      v2 = vertex[j, v[1], :]
      v3 = vertex[j, v[2], :]
      Q = assemble_face(v1, v2, v3).dot(d_inv_mean[i])
      dets.append(np.linalg.det(Q))
      Q.shape = (9, 1)
      deform[j, i, :] = Q.flat
  dets = np.array(dets).reshape(utils.F_NUM, vertex.shape[0])
  np.save(open(os.path.join(DATA_DIR, "%s_dets.npy"%label), "wb"), dets)
  np.save(open(os.path.join(DATA_DIR, "%s_d_inv_mean.npy"%label), "wb"), d_inv_mean)
  np.save(open(os.path.join(DATA_DIR, "%s_deform.npy"%label), "wb"), deform)
  mean_deform = np.array(deform.mean(axis=0))
  mean_deform.shape = (utils.F_NUM, 9)
  std_deform = np.array(deform.std(axis=0))
  std_deform.shape = (utils.F_NUM, 9)
  np.save(open(os.path.join(DATA_DIR, "%s_mean_deform.npy"%label), "wb"), mean_deform)
  np.save(open(os.path.join(DATA_DIR, "%s_std_deform.npy"%label), "wb"), std_deform)
  print('\n [**] finish load_deformation of %s in %fs' % (label, time.time() - start))
  return[d_inv_mean, deform, dets, mean_deform, std_deform]

# calculating the inverse of mean vertex matrix, v^-1
def get_inv_mean(mean_vertex, facet):
  print(" [**] begin get_inv_mean ...")
  start = time.time()
  d_inv_mean = np.zeros((utils.F_NUM, 3, 3))
  for i in range(0, utils.F_NUM):
    v = [j - 1 for j in facet[i, :]]
    v1 = mean_vertex[v[0], :]
    v2 = mean_vertex[v[1], :]
    v3 = mean_vertex[v[2], :]
    d_inv_mean[i] = assemble_face(v1, v2, v3)
    d_inv_mean[i] = np.linalg.inv(d_inv_mean[i])
  print(' [**] finish get_inv_mean in %fs' % (time.time() - start))
  return d_inv_mean

# import the 4th point of the triangle, and calculate the deformation
def assemble_face(v1, v2, v3):
  v21 = np.array((v2 - v1))
  v31 = np.array((v3 - v1))
  v41 = np.cross(list(v21.flat), list(v31.flat))
  v41 /= np.sqrt(np.linalg.norm(v41))
  return np.column_stack((v21, np.column_stack((v31, v41))))

# build sparse matrix
def build_equation(m_datas, basis_num):
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

# calculating vertex-based presentation(PCA) using t-vertex
def get_v_basis(vertex, label="female"):
  print(" [**] begin get_v_basis of %s ..."%label)
  start = time.time()
  body_num = vertex.shape[0]
  mean_vertex = np.array(vertex.mean(axis=0)).reshape(utils.V_NUM, 3)
  vertex -= mean_vertex
  std_vertex = np.array(vertex.std(axis=0)).reshape(utils.V_NUM, 3)
  vertex /= std_vertex
  vertex.shape = (vertex.shape[0], 3 * utils.V_NUM)
  v = vertex.transpose()
  # principle component analysis
  v_basis, v_sigma, V = np.linalg.svd(v, full_matrices=0)
  v_basis = np.array(v_basis[:, :utils.V_BASIS_NUM]).reshape(3 * utils.V_NUM, utils.V_BASIS_NUM)
  np.save(open(os.path.join(DATA_DIR, "%s_v_basis.npy"%label), "wb"), v_basis)
  # coefficient
  v_coeff = np.dot(v_basis.transpose(), v)
  v_pca_mean = np.array(np.mean(v_coeff, axis=1))
  v_pca_mean.shape = (v_pca_mean.size, 1)
  v_pca_std = np.array(np.std(v_coeff, axis=1))
  v_pca_std.shape = (v_pca_std.size, 1)
  vertex.shape = (body_num, utils.V_NUM, 3)
  vertex *= std_vertex
  vertex += mean_vertex
  np.save(open(os.path.join(DATA_DIR, "%s_v_basis.npy"%label),"wb"), v_basis)
  np.save(open(os.path.join(DATA_DIR, "%s_v_coeff.npy"%label),"wb"), v_coeff)
  print(' [**] finish get_v_basis in %fs' % (time.time() - start))
  return [v_basis, v_coeff, v_pca_mean, v_pca_std]

# calculating deform-based presentation(PCA)
def get_d_basis(deform, label="female"):
  print(" [**] begin get_d_basis of %s..."%label)
  start = time.time()
  body_num = deform.shape[0]
  mean_deform = np.array(deform.mean(axis=0))
  mean_deform.shape = (utils.F_NUM, 9)
  std_deform = np.array(deform.std(axis=0))
  std_deform.shape = (utils.F_NUM, 9)
  deform -= mean_deform
  deform /= std_deform
  deform.shape = (deform.shape[0], 9 * utils.F_NUM)
  d = deform.transpose()

  # principle component analysis
  d_basis, d_sigma, V = np.linalg.svd(d, full_matrices=0)
  d_basis = np.array(d_basis[:, :utils.D_BASIS_NUM]).reshape(9 * utils.F_NUM, utils.D_BASIS_NUM)

  d_coeff = np.dot(d_basis.transpose(), d)
  d_pca_mean = np.array(np.mean(d_coeff, axis=1))
  d_pca_mean.shape = (d_pca_mean.size, 1)
  d_pca_std = np.array(np.std(d_coeff, axis=1))
  d_pca_std.shape = (d_pca_std.size, 1)

  np.save(open(os.path.join(DATA_DIR, "%s_d_basis.npy"%label), "wb"), d_basis)
  np.save(open(os.path.join(DATA_DIR, "%s_d_coeff.npy"%label), "wb"), d_coeff)
  deform.shape = (body_num, utils.F_NUM, 9)
  deform *= std_deform
  deform += mean_deform
  print(' [**] finish get_d_basis of %s in %fs' % (label, time.time() - start))
  return [d_basis, d_coeff, d_pca_mean, d_pca_std]

# cosntruct the related matrix A to change deformation into vertex
def get_d2v_matrix(d_inv_mean, facet, label="female"):
  print(' [**] begin reload A&lu maxtrix of %s'%label)
  start = time.time()
  data = []
  rowidx = []
  colidx = []
  r = 0
  off = utils.V_NUM * 3
  shape = (utils.F_NUM * 9, (utils.V_NUM + utils.F_NUM) * 3)
  for i in range(0, utils.F_NUM):
    coeff = construct_coeff_mat(d_inv_mean[i])
    v = [c - 1 for c in facet[i, :]]
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
  d2v = scipy.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
  np.savez(os.path.join(DATA_DIR, "%s_d2v"%label), row=d2v.row,
    col=d2v.col,data=d2v.data, shape=d2v.shape)
  lu = scipy.sparse.linalg.splu(d2v.transpose().dot(d2v).tocsc())
  print(' [**] finish load A&lu of %s in %fs.' % (label, time.time() - start))
  return [d2v, lu]

# construct the matrix = v_mean_inv.dot(the matrix consists of 0 -1...)
def construct_coeff_mat(mat):
  tmp = -mat.sum(0)
  return np.row_stack((tmp, mat)).transpose()

# calculate the mapping matrix from measures to vertex-based
def get_m2v(v_coeff, t_measure, label="female"):
  print(' [**] begin load_m2v of %s... '%label)
  start = time.time()
  V = v_coeff.copy()
  V.shape = (V.size, 1)
  M = build_equation(t_measure, utils.V_BASIS_NUM)
  # solve transform matrix
  MtM = M.transpose().dot(M)
  MtV = M.transpose().dot(V)
  ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtV))
  ans.shape = (utils.V_BASIS_NUM, utils.M_NUM)
  np.save(open(os.path.join(DATA_DIR, "%s_m2v.npy"%label), "wb"), ans)
  print(' [**] finish get_m2v of %s in %fs' % (label, time.time() - start))
  return ans

# calculate global mapping from measure to deformation PCA coeff
def get_m2d(d_coeff, t_measure, label="female"):
  print(' [**] begin load_m2d of %s... '%label)
  start = time.time()
  D = d_coeff.copy()
  D.shape = (D.size, 1)
  M = build_equation(t_measure, utils.D_BASIS_NUM)
  # solve transform matrix
  MtM = M.transpose().dot(M)
  MtD = M.transpose().dot(D)
  ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtD))
  ans.shape = (utils.D_BASIS_NUM, utils.M_NUM)
  np.save(open(os.path.join(DATA_DIR, "%s_m2d.npy"%label), "wb"), ans)
  print(' [**] finish load_m2d of %s in %fs' % (label, time.time() - start))
  return ans

# get color dict & mask
def get_map(facet):
  tmp = open(os.path.join(DATA_DIR, 'body_part.obj'), 'r').read()
  tmp = tmp[tmp.index('\nv'): tmp.index("\n#!") - 1].replace('v', '')
  tmp = list(map(float, tmp.replace('\n', ' ').split()))
  body = np.array(tmp).reshape(utils.V_NUM, 6)
  body = np.array(body[:, 3:])
  color_list = []
  for i in range(0, utils.V_NUM):
    color_list.append((body[i, 0], body[i, 1], body[i, 2]))
  mask = np.zeros((19, utils.F_NUM), dtype=bool)
  for i in range(0, utils.F_NUM):
    sys.stdout.write('\r>> get map of facet %d'%(i))
    sys.stdout.flush()
    v = facet[i, :] - 1
    tmp = set()
    for j in v:
      c = utils.PART.index(color_list[j])
      for k in utils.P2M[c]:
        tmp.add(k)
    for j in tmp:
      mask[j, i] = 1
  np.save(open(os.path.join(DATA_DIR, "mask.npy"), "wb"), mask)
  return [utils.PART, mask]

# local map matrix: measure->deform
def local_matrix(mask, deform, measure, label="female"):
  print(' [**] begin solve local_matrix of %s'%label)
  start = time.time()
  L_tosave = []
  body_num = deform.shape[0]
  for i in range(0, utils.F_NUM):
    sys.stdout.write('\r>> calc local map of %s NO.%d'%(label, i))
    sys.stdout.flush()
    S = np.array(deform[:, i, :])
    S.shape = (S.size, 1)
    t_mask = np.array(mask[:, i])
    t_mask.shape = (utils.M_NUM, 1)
    t_mask = t_mask.repeat(body_num, axis=1)
    m = np.array(measure[t_mask])
    m.shape = (m.size // body_num, body_num)
    M = build_equation(m, 9)
    # solve transform matrix
    MtM = M.transpose().dot(M)
    MtS = M.transpose().dot(S)
    ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtS))
    ans.shape = (9, m.size // body_num)
    L_tosave.append(list(ans))
  np.save(open(os.path.join(DATA_DIR, "%s_local.npy"%label), "wb"), L_tosave)
  print('\n [**] finish solve local_matrix of %s in %fs' % (label, time.time() - start))

# calculate relationship directly
def rfe_local(dets, deform, measure, label="female", k_features=9):
  print(' [**] begin rfe_local of %s'%label)
  start = time.time()
  body_num = deform.shape[0]
  mean_measure = np.array(measure.mean(axis=1)).reshape(utils.M_NUM, 1)
  std_measure = np.array(measure.std(axis=1)).reshape(utils.M_NUM, 1)
  t_measure = measure - mean_measure
  t_measure /= std_measure
  x = t_measure.transpose()

  pool = Pool(processes = 8)
  tasks = [(i, dets[i,:], deform[:,i,:], body_num, x, measure, k_features) for i in range(utils.F_NUM)]
  results = pool.starmap(rfe_multiprocess, tasks)
  pool.close()
  pool.join()

  rfe_mat = np.array([ele[0] for ele in results]).reshape(utils.F_NUM, 9, k_features)
  mask = np.array([ele[1] for ele in results]).reshape(utils.F_NUM, utils.M_NUM).transpose()

  np.save(open(os.path.join(DATA_DIR, "%s_rfemat.npy"%label), "wb"), rfe_mat)
  np.save(open(os.path.join(DATA_DIR, "%s_rfemask.npy"%label), "wb"), mask)
  print("[**] finish rfe_mat calc of %s in %fs"%(label, time.time()-start))
  return [dets, mask, rfe_mat]

def rfe_multiprocess(i, dets, deform, body_num, x, measure, k_features):
  sys.stdout.write('>> calc rfe map NO.%d\n'%(i))
  y = np.array(dets).reshape(body_num, 1)
  model = LinearRegression()
  # recurcive feature elimination
  rfe = RFE(model, k_features)
  rfe.fit(x, y.ravel())
  # mask.append(rfe.support_)
  flag = np.array(rfe.support_).reshape(utils.M_NUM, 1)
  flag = flag.repeat(body_num, axis=1)

  # calculte linear mapping mat
  S = np.array(deform)
  S.shape = (S.size, 1)
  m = np.array(measure[flag])
  m.shape = (k_features, body_num)
  M = build_equation(m, 9)
  MtM = M.transpose().dot(M)
  MtS = M.transpose().dot(S)
  ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtS))
  ans.shape = (9, k_features)
  return [ans, rfe.support_]

def random_multiprocess(i, dets, deform, body_num, x, measure, k_features):
  sys.stdout.write('>> calc random map NO.%d\n'%(i))
  # recurcive feature elimination
  flag = np.zeros((19, 1), dtype=bool)
  index = random.sample(range(19), k_features)
  for i in index:
    flag[i,0] = True
  F = flag.repeat(body_num, axis=1)

  # calculte linear mapping mat
  S = np.array(deform)
  S.shape = (S.size, 1)
  m = np.array(measure[F])
  m.shape = (k_features, body_num)
  M = build_equation(m, 9)
  MtM = M.transpose().dot(M)
  MtS = M.transpose().dot(S)
  ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtS))
  ans.shape = (9, k_features)
  return [ans, flag]

# ===========================================================================

# train all data
def train():
  genders = ["female"]#, "male"]
  for gender in genders:
    # generate and load control point from txt to npy file
    cp = convert_cp()
    facet = convert_template()
    vertex = obj2npy(gender)[0]
    d_inv_mean, deform, dets, _, _ = load_d_data(vertex, facet, label=gender)
    measure = convert_measure(cp, vertex, facet, label=gender)[0]
    d_basis, d_coeff, _, _ = get_d_basis(deform, label=gender)
    # v_basis, v_coeff, _, _, = get_v_basis(deform, label=gender)
    rfe_local(dets, deform, measure, label=gender, k_features=9)


if __name__ == "__main__":
  train()
