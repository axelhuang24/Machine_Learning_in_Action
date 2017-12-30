'''
Created on Jun 1, 2011

@author: Kai HUANG
'''
from numpy import *

# function: load DataSet
# fileName(文件名), delim(分隔符)
# return: dataSet
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

# function: PCA简化数据
# input: dataMat(数据集), topNfeat(想要的数据特征的个数)
# return: lowDDataMat(降维之后的数据集), reconMat(重建的原数据集)
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

# function: 用平均值填充缺失值
# input: none
# return: dataSet	
def replaceNanWithMean():
    datMat = loadtxt('secom.data')
    #datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i]))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i]))[0],i] = meanVal  #set NaN values to mean
    return datMat