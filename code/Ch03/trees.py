'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import numpy as np
import random
# def createDataSet():
#     dataSet = [[1, 1, 'yes'],
#                [1, 1, 'yes'],
#                [1, 0, 'no'],
#                [0, 1, 'no'],
#                [0, 1, 'no']]
#     labels = ['no surfacing','flippers']
#     #change to discrete values
#     return dataSet, labels

##############################
def createDataSet():
    dataSet = [[1, 0.1, 1, 'no'],
               [1, 0.3, 1, 'yes'],
               [1, 0.4, 0, 'no'],
               [0, 0.6, 1, 'yes'],
               [0, 1.7, 1, 'yes'],
               [1, 1.8, 1, 'yes'],
               [1, 0.35, 1, 'no'],
               [0, 1.85, 0, 'no']]
    labels = ['no surfacing','continue_feature','flippers']
    #change to discrete values
    return dataSet, labels
##########################

def calcShannonEnt(dataSet):
    '''
    计算传入数据集的熵，只按yes or no进行二分类
    :param dataSet: 传入数据集
    :return:
    '''
    numEntries = len(dataSet)
    labelCounts = {}  # 键为label，值为数据集中某个label的样本数目，这里的label是yes no
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt    #这个返回值是传入数据集的熵
    
def splitDataSet(dataSet, axis, value=None):
    '''
    传入一个数据集，把axis维度等于value的数据拿出来，去掉axis维度，再返回
    :param dataSet: 数据集
    :param axis: 指定的维度下标
    :param value: 维度取值
    :return: 数据集
    '''
    if isinstance(dataSet[0][axis],(float, np.float16, np.float32, np.float64)):
        retDataSet = []
        for featVec in dataSet:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
        return retDataSet

    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
#     baseEntropy = calcShannonEnt(dataSet)
#     bestInfoGain = 0.0; bestFeature = -1
#     for i in range(numFeatures):        # iterate over all the features
#         featList = [example[i] for example in dataSet]# create a list of all the examples of this feature
#         uniqueVals = set(featList)       # get a set of unique values
#         newEntropy = 0.0
#         for value in uniqueVals:
#             subDataSet = splitDataSet(dataSet, i, value)
#             prob = len(subDataSet)/float(len(dataSet))
#             newEntropy += prob * calcShannonEnt(subDataSet)
#         infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
#         if (infoGain > bestInfoGain):       #compare this to the best gain so far
#             bestInfoGain = infoGain         #if better than current best, set to best
#             bestFeature = i
#     return bestFeature                      #returns an integer

########################################################################
def chooseBestFeatureToSplit(dataSet):
    dataSet_length = len(dataSet)
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    float_types = (float, np.float16, np.float32, np.float64)

    ls_continue = []
    for i in range(numFeatures):  # iterate over all the features
        if isinstance(dataSet[0][i], float_types):
            dich_entro_list = []
            # dichotomy = -1
            sorted_dataSet = sorted(dataSet, key=lambda sample: sample[i])
            for dichotomy in range(dataSet_length - 1):
                D_minus = sorted_dataSet[:dichotomy + 1]
                weight_minus = (dichotomy + 1) / dataSet_length
                D_plus = sorted_dataSet[dichotomy + 1:]
                weight_plus = 1 - weight_minus
                entropy = weight_minus * calcShannonEnt(D_minus) + weight_plus * calcShannonEnt(D_plus)
                dich_entro_list.append((dichotomy, entropy))  # 列表，其中每个元素是(二分位置截至坐标，熵)
            # 按熵对列表升序排序
            sorted_dich_entro_list = sorted(dich_entro_list, key=lambda sample: sample[1])
            # 取出最小熵
            newEntropy = sorted_dich_entro_list[0][1]
            # 其中每个元素是(dichotomy, entropy,i)
            ls_continue.append((*sorted_dich_entro_list[0], i))

        else:
            featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
            uniqueVals = set(featList)  # get a set of unique values
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    if isinstance(dataSet[0][bestFeature], float_types):
        triplet = [t for t in ls_continue if t[2] == bestFeature][0]
        return triplet # 返回值形式为(二分位置截至坐标，熵，连续属性下标)
    else:
        return bestFeature  # returns an integer

#####################################################################


def majorityCnt(classList):
    '''
    返回类别数最多的label的值
    :param classList:
    :return:
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]
#################################################
def createTree(dataSet,labels,threshold=1):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] # stop splitting when all of the classes are equal
    if (len(dataSet[0]) == 1) or (len(classList)<=threshold): #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    if isinstance(bestFeat, tuple):
        dichotomy, _, index = bestFeat
        bestFeatLabel = labels[index]
        myTree = {bestFeatLabel: {}}
        sorted_dataSet = sorted(dataSet, key=lambda sample: sample[index])
        data_minus = sorted_dataSet[:dichotomy+1]
        del (labels[index])
        subLabels = labels[:]
        myTree[bestFeatLabel][f"<={sorted_dataSet[dichotomy][index]}"] = createTree(splitDataSet(data_minus,index),subLabels,threshold)
        data_plus = sorted_dataSet[dichotomy+1:]
        subLabels = labels[:]
        myTree[bestFeatLabel][f">{sorted_dataSet[dichotomy][index]}"] = createTree(splitDataSet(data_plus,index),subLabels,threshold)

    else:
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]  #copy all of labels, so trees don't mess up existing labels
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),\
                                                      subLabels,threshold)
    return myTree
##############################################
# def createTree(dataSet,labels):
#     classList = [example[-1] for example in dataSet]
#     if classList.count(classList[0]) == len(classList):
#         return classList[0] # stop splitting when all of the classes are equal
#     if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)
#     bestFeatLabel = labels[bestFeat]
#     myTree = {bestFeatLabel:{}}
#     del(labels[bestFeat])
#     featValues = [example[bestFeat] for example in dataSet]
#     uniqueVals = set(featValues)
#     for value in uniqueVals:
#         subLabels = labels[:]  #copy all of labels, so trees don't mess up existing labels
#         myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),\
#                                                   subLabels)
#     return myTree

# 可以加个batch的逻辑，batch size为1也能处理
def classify(inputTree,featLabels,testVec):
    '''
    :param inputTree: {'continue_feature': {'<=0.4': 'no', '>0.4': 'yes'}}
    :param featLabels: ['no surfacing','continue_feature','flippers']
    :param testVec:[0, 1.85, 0, 'yes']
    :return:
    '''
    firstStr = list(inputTree.keys())[0] # 'continue_feature'
    secondDict = inputTree[firstStr] # {'<=0.4': 'no', '>0.4': 'yes'}
    featIndex = featLabels.index(firstStr)  # 1
    key = testVec[featIndex]  # 1.85
    for condition,value in secondDict.items():
        if isinstance(condition,str):
            # featIndex = featLabels.index(firstStr)  # 1
            # key = testVec[featIndex]  # 1.85
            if eval(f"{key} {condition}"):
                valueOfFeat = secondDict[condition]
                if isinstance(valueOfFeat, dict):
                    classLabel = classify(valueOfFeat, featLabels, testVec)
                else:
                    classLabel = valueOfFeat
                return classLabel
        else:
            break
    featIndex = featLabels.index(firstStr) # 1
    key = testVec[featIndex] # 1.85
    # valueOfFeat = secondDict[key] # no

    if key in secondDict:
        valueOfFeat = secondDict[key]
    else:
        # 如果key不在secondDict中，则随机选择一个键
        random_key = random.choice(list(secondDict.keys()))
        valueOfFeat = secondDict[random_key]

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
