#!/usr/bin/python
# coding=utf-8

import sys
sys.path.append("..")
from dataProcess.rawData import *
from openpyxl import Workbook
from openpyxl import load_workbook
from fancyimpute import KNN, SoftImpute, NuclearNormMinimization, MICE, SimpleFill
from fancyimpute import MatrixFactorization
import numpy as np
import numpy.matlib
import time


class measureMining:
    '''
        A measureMining is responsible for mining the correlation between measure data human
        input:  all original data containing: measure...
        output: all correlation between measures
        usage:  given part of measures, and predict other unknown data
    '''
    __metaclass__ = Singleton

    def __init__(self, data):
        # load all necessary data
        self.test_size = 300
        self.data = data
        self.GIRTH = np.array(self.data.paras["GIRTH"], dtype=bool).reshape(
            self.data.measure_num, 1)
        self.LENGTH = np.array(self.data.paras["LENGTH"], dtype=bool).reshape(
            self.data.measure_num, 1)
        self.filePath = data.paras['dataPath'] + "NPYdata/measureData/"
        self.m2m = list(self.data.paras['m2m'])
        [self.m_basis, self.m_sigma, self.m_pca_mean, self.m_pca_std, self.m_coeff] = \
            self.get_measure_basis()
        self.cfTable = self.itemCF()

        # , "SoftImpute",   "Nuclear", "Matrix" ]#, "Similarity"]
        self.impute_name = ["SimpleFill", "KNN", "MICE"]
        self.impute_method = [SimpleFill(), KNN(), MICE()]  # ,
        # SoftImpute(5), NuclearNormMinimization(),
        # MatrixFactorization(learning_rate=0.01),
        # SimilarityWeightedAveraging()]

    # -------------------------------------------------------
    '''calculating measure-based presentation(PCA)'''
    # -------------------------------------------------------

    def get_measure_basis(self):
        print(" [**] begin get_measure_basis ...")
        m_basis_file = self.filePath + 'm_basis.npy'
        m_sigma_file = self.filePath + 'm_sigma.npy'
        m_pca_mean_file = self.filePath + 'm_pca_mean.npy'
        m_pca_std_file = self.filePath + 'm_pca_std.npy'
        m_coeff_file = self.filePath + 'm_coeff.npy'
        start = time.time()
        if self.data.paras["reload_measure_basis"]:
            # principle component analysis
            m_basis, m_sigma, M = np.linalg.svd(
                self.data.t_measures, full_matrices=0)
            m_coeff = np.dot(m_basis.transpose(), self.data.t_measures)
            m_pca_mean = np.array(np.mean(m_coeff, axis=1))
            m_pca_mean.shape = (m_pca_mean.size, 1)
            m_pca_std = np.array(np.std(m_coeff, axis=1))
            m_pca_std.shape = (m_pca_std.size, 1)
            m_basis = np.array(m_basis)
            m_coeff = np.array(m_coeff).reshape(
                m_coeff.shape[0], self.data.body_count)
            self.data.save_NPY([m_basis_file, m_sigma_file, m_pca_mean_file, m_pca_std_file, m_coeff_file],
                               [m_basis, m_sigma, m_pca_mean, m_pca_std, m_coeff])
            print(' [**] finish get_vertex_basis in %fs' %
                  (time.time() - start))
            return [m_basis, m_sigma, m_pca_mean, m_pca_std, m_coeff]
        else:
            print(' [**] finish get_vertex_basis in %fs' %
                  (time.time() - start))
            return self.data.load_NPY([m_basis_file, m_sigma_file, m_pca_mean_file, m_pca_std_file, m_coeff_file])

    # ------------------------------------------------------------------------------------------------------
    '''calculate the similarity between each measure, using Pearson Correlation, training data(300-1531) '''
    # ------------------------------------------------------------------------------------------------------

    def itemCF(self):
        print(' [**] begin get similarity of measures')
        start = time.time()
        table = np.matlib.zeros((self.data.measure_num, self.data.measure_num))
        if self.data.paras['reload_cfTable']:
            # calculation
            for i in range(0, self.data.measure_num):
                table[i, i] = 0
                for j in range(i + 1, self.data.measure_num):
                    a = np.array(self.data.t_measures[i, self.test_size:]).reshape(
                        1, self.data.body_count - self.test_size)
                    b = np.array(self.data.t_measures[j, self.test_size:]).reshape(
                        self.data.body_count - self.test_size, 1)
                    similar = (a.dot(b) / (np.linalg.norm(a)
                                           * np.linalg.norm(b)))[0, 0]
                    table[i, j] = table[j, i] = similar
            # save the result in excel
            wb = Workbook()
            ws = wb.get_active_sheet()
            top = self.data.measure_num
            for i in range(0, top):
                ws.cell(row=i * (top + 2) + 1, column=1).value = i
                ws.cell(row=i * (top + 2) + 1,
                        column=2).value = self.data.measure_str[i]
                for j in range(0, top):
                    ws.cell(row=i * (top + 2) + j + 2,
                            column=1).value = self.data.measure_str[j]
                    ws.cell(row=i * (top + 2) + j + 2,
                            column=2).value = table[i, j]
            wb.save(filename=self.data.paras['ansPath'] + 'similarity.xlsx')
            self.data.save_NPY([self.filePath + "cfTable.npy"], [table])
            print(' [**] finish get and save cf of measures in %s s.' %
                  (time.time() - start))
            return table
        else:
            print(' [**] finish get and save cf of measures in %s s.' %
                  (time.time() - start))
            return self.data.load_NPY([self.filePath + "cfTable.npy"])[0]

    # -------------------------------------------------------------------------------------------------
    '''given flag(means which value is available), value, predict completed measures, simple sum all'''
    # -------------------------------------------------------------------------------------------------

    def simpleAverage(self, flag, data, dist):
        data = np.array(data).reshape(self.data.measure_num, 1)
        output = flag * data

        for j in range(0, output.shape[0]):
            if output[j, 0] == 0:
                coeff = np.array(self.cfTable[j, :]).reshape(
                    self.data.measure_num, 1)
                if (abs(coeff).sum() == 0):
                    output[j, 0] = 0
                else:
                    if dist & ((self.GIRTH[j, 0] | self.LENGTH[j, 0]) != False):
                        if self.GIRTH[j, 0]:
                            x = data[flag & self.GIRTH]
                            coeff = coeff[flag & self.GIRTH]
                        else:
                            x = data[flag & self.LENGTH]
                            coeff = coeff[flag & self.LENGTH]
                        x.shape = (1, x.size)
                    else:
                        x = data[flag]
                        x.shape = (1, x.size)
                        coeff = coeff[flag]
                    output[j, 0] = x.dot(coeff) / (abs(coeff).sum())
        return output

    # -------------------------------------------------------------------------------------------------
    '''Predict, let small smaller, big bigger'''
    # -------------------------------------------------------------------------------------------------

    def caseAmplify(self, flag, data, dist):
        frac = 2.5
        data = np.array(data).reshape(self.data.measure_num, 1)
        output = flag * data

        for j in range(0, output.shape[0]):
            if output[j, 0] == 0:
                coeff = np.array(self.cfTable[j, :]).reshape(
                    self.data.measure_num, 1)
                if (abs(coeff).sum() == 0):
                    output[j, 0] = 0
                else:
                    if dist & ((self.GIRTH[j, 0] | self.LENGTH[j, 0]) != False):
                        if self.GIRTH[j, 0]:
                            x = data[flag & self.GIRTH]
                            coeff = coeff[flag & self.GIRTH]
                        else:
                            x = data[flag & self.LENGTH]
                            coeff = coeff[flag & self.LENGTH]
                        x.shape = (1, x.size)
                    else:
                        x = data[flag]
                        x.shape = (1, x.size)
                        coeff = coeff[flag]
                    tmp = coeff * (abs(coeff)**1.5)
                    output[j, 0] = x.dot(tmp) / (abs(tmp).sum())
        return output

    # -------------------------------------------------------------------------------------------------
    '''using imputation for missing data'''
    # -------------------------------------------------------------------------------------------------

    def Imputate(self, flag, data, dist, imputor):
        output = np.array(data).reshape(self.data.measure_num, 1)
        output[~flag] = np.nan
        tmp = np.array(self.data.t_measures[:, self.test_size:])
        tmp = np.column_stack((tmp, output)).transpose()
        tmp = imputor.complete(tmp)
        output = np.array(tmp[-1, :]).reshape(self.data.measure_num, 1)
        return output

    # -------------------------------------------------------------------------------------
    '''given flag(means which value is available), value, predict completed measures'''
    # -------------------------------------------------------------------------------------

    def getPredict(self, flag, data):
        if ((flag == True).sum() == self.data.measure_num):
            return data
        else:
            solver = MICE()
            return self.Imputate(flag, data, True, solver)

    def spetialTest(self):
        print(' [**] Spetial Test...')
        table_size = 26
        input_file = self.filePath + "input.xlsx"
        input_wb = load_workbook(filename=input_file)
        sheets = input_wb.get_sheet_names()
        input_ws = input_wb.get_sheet_by_name(sheets[0])
        output_file = '../../result/prediction.xlsx'
        output_wb = Workbook()
        output_ws = output_wb.get_active_sheet()
        # read test data
        test_data = []
        for i in range(0, 8):
            flag = np.zeros((self.data.measure_num, 1), dtype=bool)
            for j in range(0, self.data.measure_num):
                flag[j, 0] = input_ws.cell(row=2 + j, column=2 + i).value
            test_data.append(flag)

        # begin test, using 0-300 data, output the offset to mean
        for i in range(0, len(test_data)):
            output_ws.cell(row=(i * table_size) + 2, column=1).value = i
            for j in range(0, self.data.measure_num):
                output_ws.cell(row=(i * table_size) + 3 + j,
                               column=1).value = self.data.measure_str[j]
            output_ws.cell(row=(i * table_size) + 22,
                           column=1).value = 'MEAN GIRTH'
            output_ws.cell(row=(i * table_size) + 23,
                           column=1).value = 'MEAN LENGTH'
            output_ws.cell(row=(i * table_size) + 24, column=1).value = 'MEAN'
            k = 2

            flag = test_data[i].copy().repeat(self.test_size, axis=1)
            # test for imputation technology
            for m in range(0, len(self.impute_method)):
                output_ws.cell(row=(i * table_size) + 2,
                               column=k).value = self.impute_name[m]
                print('Experiment %d, ' % i, self.impute_name[m])
                input = self.data.t_measures.copy()
                input[:, :self.test_size][~flag] = np.nan
                tmp = self.impute_method[m].complete(input.transpose())
                output = tmp.transpose()[:, :self.test_size]
                input = self.data.t_measures[:, :self.test_size]
                error = np.array(np.mean(abs(output - input), axis=1)
                                 ).reshape(self.data.measure_num, 1)
                print(error)
                # error = error * self.data.std_measures
                for j in range(0, self.data.measure_num):
                    output_ws.cell(row=(i * table_size) + 3 + j,
                                   column=k).value = error[j, 0]
                output_ws.cell(row=(i * table_size) + 22, column=k).value = np.average(
                    abs(error[(~test_data[i]) & self.GIRTH]))
                output_ws.cell(row=(i * table_size) + 23, column=k).value = np.average(
                    abs(error[(~test_data[i]) & self.LENGTH]))
                output_ws.cell(row=(
                    i * table_size) + 24, column=k).value = np.average(abs(error[(~test_data[i])]))
                k = k + 1
            output_wb.save(output_file)
        output_wb.save(output_file)
        print(' [**] Finish Spetial Test...')

    # -------------------------------------------------------------------------------------
    '''test all methods'''
    # -------------------------------------------------------------------------------------

    def test(self):
        print(' [**] Begin predict input data...')
        table_size = 26
        input_file = self.filePath + "input.xlsx"
        input_wb = load_workbook(filename=input_file)
        sheets = input_wb.get_sheet_names()
        input_ws = input_wb.get_sheet_by_name(sheets[0])
        output_file = '../../result/prediction.xlsx'
        output_wb = Workbook()
        output_ws = output_wb.get_active_sheet()

        # read test data
        test_data = []
        for i in range(0, 5):
            flag = np.zeros((self.data.measure_num, 1), dtype=bool)
            for j in range(0, self.data.measure_num):
                flag[j, 0] = input_ws.cell(row=2 + j, column=2 + i).value
            test_data.append(flag)

        # begin test, using 0-300 data, output the offset to mean
        for i in range(0, len(test_data)):
            output_ws.cell(row=(i * table_size) + 2, column=1).value = i
            for j in range(0, self.data.measure_num):
                output_ws.cell(row=(i * table_size) + 3 + j,
                               column=1).value = self.data.measure_str[j]
            output_ws.cell(row=(i * table_size) + 22,
                           column=1).value = 'MEAN GIRTH'
            output_ws.cell(row=(i * table_size) + 23,
                           column=1).value = 'MEAN LENGTH'
            output_ws.cell(row=(i * table_size) + 24, column=1).value = 'MEAN'
            k = 2

            # test for imputation technology
            for m in range(0, len(self.impute_method)):
                output_ws.cell(row=(i * table_size) + 2,
                               column=k).value = self.impute_name[m]
                error = np.zeros((self.data.measure_num, 1))
                for j in range(0, self.test_size):
                    print('Experiment %d, ' %
                          i, self.impute_name[m], ' sample: ', j)
                    input = np.array(self.data.t_measures[:, j]).reshape(
                        self.data.measure_num, 1)
                    output = self.Imputate(
                        test_data[i], input, False, self.impute_method[m])
                    error += abs(output - input)
                error /= self.test_size
                error *= self.data.std_measures
                for j in range(0, self.data.measure_num):
                    output_ws.cell(row=(i * table_size) + 3 + j,
                                   column=k).value = error[j, 0]
                output_ws.cell(row=(i * table_size) + 22, column=k).value = np.average(
                    abs(error[(~test_data[i]) & self.GIRTH]))
                output_ws.cell(row=(i * table_size) + 23, column=k).value = np.average(
                    abs(error[(~test_data[i]) & self.LENGTH]))
                output_ws.cell(row=(
                    i * table_size) + 24, column=k).value = np.average(abs(error[(~test_data[i])]))
                k = k + 1

            # test for simple weighted without classification
            output_ws.cell(row=(i * table_size) + 2,
                           column=k).value = 'simple weighted'
            error = np.zeros((self.data.measure_num, 1))
            for j in range(0, self.test_size):
                input = np.array(self.data.t_measures[:, j]).reshape(
                    self.data.measure_num, 1)
                output = self.simpleAverage(test_data[i], input, False)
                error += abs(output - input)
            error /= self.test_size
            error *= self.data.std_measures
            for j in range(0, self.data.measure_num):
                output_ws.cell(row=(i * table_size) + 3 + j,
                               column=k).value = error[j, 0]
            output_ws.cell(row=(i * table_size) + 22, column=k).value = np.average(
                abs(error[(~test_data[i]) & self.GIRTH]))
            output_ws.cell(row=(i * table_size) + 23, column=k).value = np.average(
                abs(error[(~test_data[i]) & self.LENGTH]))
            output_ws.cell(row=(i * table_size) + 24,
                           column=k).value = np.average(abs(error[(~test_data[i])]))
            k += 1

            # test for case amplication(2.5) without classification
            output_ws.cell(row=(i * table_size) + 2,
                           column=k).value = 'case amplication'
            error = np.zeros((self.data.measure_num, 1))
            for j in range(0, self.test_size):
                input = np.array(self.data.t_measures[:, j]).reshape(
                    self.data.measure_num, 1)
                output = self.caseAmplify(test_data[i], input, False)
                error += abs(output - input)
            error /= self.test_size
            error *= self.data.std_measures
            for j in range(0, self.data.measure_num):
                output_ws.cell(row=(i * table_size) + 3 + j,
                               column=k).value = error[j, 0]
            output_ws.cell(row=(i * table_size) + 22, column=k).value = np.average(
                abs(error[(~test_data[i]) & self.GIRTH]))
            output_ws.cell(row=(i * table_size) + 23, column=k).value = np.average(
                abs(error[(~test_data[i]) & self.LENGTH]))
            output_ws.cell(row=(i * table_size) + 24,
                           column=k).value = np.average(abs(error[(~test_data[i])]))
            k = k + 1

            output_wb.save(output_file)
        output_wb.save(output_file)
        print(' [**] Finish predict input data...')


###############################################################################
###############################################################################
if __name__ == "__main__":
    data = rawData("../parameter.json")
    miner = measureMining(data)
    miner.test()
