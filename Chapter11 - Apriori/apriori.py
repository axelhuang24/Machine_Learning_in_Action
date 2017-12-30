'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

# function: create a dataset
# input: none
# return: a simple dataSet
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# function: 构建第一个候选项集的集合
# input: dataSet
# return: 候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict    

# function: 计算支持度
# input: D(数据集), Ck(候选相集), minSupport(最小支持度)
# return: retList(频繁集的列表), supportData(字典(频繁集: 支持度))
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for can in Ck:	
        for tid in D:    
            if can.issubset(tid):
                if not (can in ssCnt): ssCnt[can]=1
                # if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D));
    retList = []
    supportData = {}
    for key in ssCnt:
        support = float(ssCnt[key])/float(numItems)
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

# function: 构建含有k个元素的待选项集
# input: Lk(这里是含有 k-2 个元素的项集), k
# return: a list of	待选项集
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

# function: 构建项集的整体函数
# input: 数据集和最小支持度
# return: 项集L, 支持度	supportData
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = [s for s in dataSet]
    # D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# function: 关联规则生成函数
# input: L(项集), supportData(包含支持度的字典), minConf=0.7(最小置信度, 默认为 0.7)
# return: bigRuleList(freqSet-conseq, conseq, conf)
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

# function: 计算置信度
# input: freqSet(频繁项集), H(可以出现在规则右部的列表H), supportData(包含支持度的字典), brl(bigRuleList), minConf(最小置信度, 默认为 0.7)
# return: pruned 之后的 H
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# function: 规则合并
# input: freqSet(频繁项集), H(可以出现在规则右部的列表H), supportData(包含支持度的字典), brl(bigRuleList), minConf(最小置信度, 默认为 0.7)
# return: none
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()      #print a blank line
        
