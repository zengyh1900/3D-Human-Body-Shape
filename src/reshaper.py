#!/usr/bin/python
# coding=utf-8

from fancyimpute import MICE
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import sys
import os
import utils

# A Reshaper provide multiple methods for synthesizing body mesh
class Reshaper:
  def __init__(self, label="female"):
    self.label_ = label
    self.facets = np.load(open(os.path.join(utils.MODEL_DIR, "facets.npy"), "rb"))
    self.normals = np.load(open(os.path.join(utils.MODEL_DIR, "normals.npy"), "rb"))
    self.mask = np.load(open(os.path.join(utils.MODEL_DIR, "mask.npy"), "rb"))
    self.rfemask = np.load(open(os.path.join(utils.MODEL_DIR, "%s_rfemask.npy"%label), "rb"))
    self.rfemat = np.load(open(os.path.join(utils.MODEL_DIR, "%s_rfemat.npy"%label), "rb"))
    self.m2d = np.load(open(os.path.join(utils.MODEL_DIR, "%s_m2d.npy"%label), "rb"))
    self.d_basis = np.load(open(os.path.join(utils.MODEL_DIR, "%s_d_basis.npy"%label), "rb"))
    self.t_measure = np.load(open(os.path.join(utils.MODEL_DIR, "%s_t_measure.npy"%label), "rb"))
    self.mean_measure = np.load(open(os.path.join(utils.MODEL_DIR, "%s_mean_measure.npy"%label), "rb"))
    self.mean_deform = np.load(open(os.path.join(utils.MODEL_DIR, "%s_mean_deform.npy"%label), "rb"))
    self.mean_vertex = np.load(open(os.path.join(utils.MODEL_DIR, "%s_mean_vertex.npy"%label), "rb"))
    self.std_measure = np.load(open(os.path.join(utils.MODEL_DIR, "%s_std_measure.npy"%label), "rb"))
    self.std_deform = np.load(open(os.path.join(utils.MODEL_DIR, "%s_std_deform.npy"%label), "rb"))
    self.cp = np.load(open(os.path.join(utils.MODEL_DIR, "cp.npy"), "rb"))

    loader = np.load(os.path.join(utils.MODEL_DIR, "%s_d2v.npz"%label))
    self.d2v = scipy.sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),shape=loader['shape'])
    self.lu = scipy.sparse.linalg.splu(self.d2v.transpose().dot(self.d2v).tocsc())
    self.local_mat = []
    tmp = np.load(open(os.path.join(utils.MODEL_DIR, "%s_local.npy"%label), "rb"))
    for i in range(0, len(tmp)):
      self.local_mat.append(np.array([c for c in tmp[i]]))

  def mapping(self, weight, flag=0):
    if flag==0:
      return self.mapping_global(weight)
    elif flag==1:
      return self.mapping_mask(weight)
    else:
      return self.mapping_rfemat(weight)

  # global mapping using t_measure
  def mapping_global(self, weight):
    weight = np.array(weight).reshape(utils.M_NUM, 1)
    weight = self.m2d.dot(weight)
    d = np.matmul(self.d_basis, weight)
    d.shape = (utils.F_NUM, 9)
    d *= self.std_deform
    d += self.mean_deform
    d.shape = (utils.F_NUM * 9, 1)
    [v, n, f] = self.d_synthesize(d)
    return [v, n, f]

  # local mapping using measure + mask
  def mapping_mask(self, weight):
    weight = np.array(weight).reshape(utils.M_NUM, 1)
    weight *= self.std_measure
    weight += self.mean_measure
    d = []
    for i in range(0, utils.F_NUM):
      mask = np.array(self.mask[:, i]).reshape(utils.M_NUM, 1)
      alpha = np.array(weight[mask])
      alpha.shape = (alpha.size, 1)
      s = self.local_mat[i].dot(alpha)
      d += [a for a in s.flat]
    d = np.array(d).reshape(utils.F_NUM * 9, 1)
    [v, n, f] = self.d_synthesize(d)
    return [v, n, f]

  # local mapping using measure + rfe_mat
  def mapping_rfemat(self, weight):
    weight = np.array(weight).reshape(utils.M_NUM, 1)
    weight *= self.std_measure
    weight += self.mean_measure
    d = []
    for i in range(0, utils.F_NUM):
      mask = np.array(self.rfemask[:, i]).reshape(utils.M_NUM, 1)
      alpha = np.array(weight[mask])
      alpha.shape = (alpha.size, 1)
      s = self.rfemat[i].dot(alpha)
      d += [a for a in s.flat]
    d = np.array(d).reshape(utils.F_NUM * 9, 1)
    [v, n, f] = self.d_synthesize(d)
    return [v, n, f]

  # synthesize a body by deform-based, given deform, output vertex
  def d_synthesize(self, deform):
    d = np.array(deform.flat).reshape(deform.size, 1)
    Atd = self.d2v.transpose().dot(d)
    x = self.lu.solve(Atd)
    x = x[:utils.V_NUM * 3]
    # move to center
    x.shape = (utils.V_NUM, 3)
    x_mean = np.mean(x, axis=0)
    x -= x_mean
    return [x, -self.normals, self.facets - 1]

  # given flag, value, predict completed measures
  def test(self, flag, data):
    if (flag == 1).sum() == self.data.m_num:
      return data
    else:
      solver = MICE()
      return self.imputate(flag, data, solver)

  # using imputation for missing data
  def get_predict(self, flag, in_data):
    output = in_data.copy()
    output.shape = (utils.M_NUM, 1)
    output[~flag] = np.nan
    solver = MICE()
    tmp = self.t_measure.copy()
    tmp = np.column_stack((tmp, output)).transpose()
    tmp = solver.complete(tmp)
    output = np.array(tmp[-1, :]).reshape(utils.M_NUM, 1)
    return output



if __name__ == "__main__":
  label = "female"
  body = Reshaper(label)
  measure = np.load(open(os.path.join(utils.MODEL_DIR, "%s_measure.npy"%label),"rb"))
  mean_measure = np.array(measure.mean(axis=1)).reshape(utils.M_NUM, 1)
  std_measure = np.array(measure.std(axis=1)).reshape(utils.M_NUM, 1)
  t_measure = measure - mean_measure
  t_measure /= std_measure

  for i in range(100):
    [v, n, f] = body.mapping(t_measure[:,i], 2)
    utils.save_obj(os.path.join(utils.MODEL_DIR, "test.obj"), v, f+1)


