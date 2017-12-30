'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

# function: load dataset
# input: file name
# return: the dataset
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
		#fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
	
#def loadDataSet(fileName):      #general function to parse tab -delimited floats
#    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
#    dataMat = [];
#    fr = open(fileName)
#    for line in fr.readlines():
#        lineArr =[]
#        curLine = line.strip().split('\t')
#        for i in range(numFeat):
#            lineArr.append(float(curLine[i]))
#        dataMat.append(lineArr)
#    return dataMat

# function: split dataset with a value of a feature
# input: dataset, the feature and the value
# return: two splited dataset
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

# function: 负责生成叶子节点
# input: dataSet 数据集
# return: the mean of the variable cible
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

# function: 计算混乱度：这里用的是(variable cible)总方差
# input: 数据集
# return: (variable cible)总方差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# function: 把数据集分成 X 和 Y, 并求得w
# input: dataset
# return: ws, X, Y	
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

# function: 建立模型树的叶子节点
# input: dataset
# return: ws	
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

# function: 计算误差
# input: dataset
# return: error	
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

# function: 用最佳方式切分数据集和生成相应的叶子节点
# input: dataSet(数据集), leafType(建立叶子节点的函数), errType(误差计算函数), ops(包含树建构所需其他参数的元组)	
# return: 切分的特征和切分的值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

# function: create the tree
# input: dataSet(数据集), leafType(建立叶子节点的函数), errType(误差计算函数), ops(包含树建构所需其他参数的元组)						  
# return: a tree(a dictionnary)
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

# function: 测试输入变量是不是一棵树
# input: obj
# return: TRUE/FALSE	
def isTree(obj):
    return (type(obj).__name__=='dict')

# function: 计算树的平均值
# input: a tree
# function: 树的平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
   
# function: 剪枝
# input: tree(树), testData(测试集)
# return: a pruned tree
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree

# function: 计算回归数叶子节点预测的值
# input: model(此时是叶子节点， 其实就是一个值), inDat(输入的数据)
# return: 预测的值	
def regTreeEval(model, inDat):
    return float(model)

# function: 计算模型数叶子节点预测的值
# input: model(此时是叶子节点， 其实就是一个线性方程), inDat(输入的数据)
# return: 预测的值
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

# function: 对单个数据点进行预测
# input: tree(决策树), inData(单个数据点), modelEval=regTreeEval(告诉我们是回归树还是决策树)
# return: 预测的值
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

# function: 对多个数据点进行预测
# input: tree(决策树), inData(多个数据点), modelEval=regTreeEval(告诉我们是回归树还是决策树)
# return: 预测的值		
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat