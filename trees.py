"""
决策树算法
@author: hegongshan https://www.hegongshan.com
"""

from math import log
import operator

def calcShannonEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVector in dataSet:
        currentLabel = featVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEntropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEntropy -= prob * log(prob,2)
    return shannonEntropy

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    """数据集中满足某一特征等于指定值的元素列表"""
    newDataSet = []
    for featVector in dataSet:
        if featVector[axis] == value:
            reducedFeatVector = featVector[:axis]
            reducedFeatVector.extend(featVector[axis+1:])
            newDataSet.append(reducedFeatVector)
    return newDataSet

def chooseBestFeatureToSplit(dataSet):
    """遍历数据集中的特征，选择最佳的数据集划分方式"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """多数表决"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 若类型完全相同，则直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时，返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value),subLabels)
    return myTree

def classify(inputTree,featureLabels,testVector):
    """决策树分类"""
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVector[featureIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featureLabels,testVector)
            else:
                classLabel = secondDict[key]
    return classLabel
    
def storeTree(inputTree,filename):
    """存储决策树"""
    import pickle
    # open(filename,'w')报如下错误：
    # TypeError: write() argument must be str, not bytes
    # pickle 默认是用二进制的形式存储数据
    # 解决办法:open(filename,'wb')
    with open(filename,'wb') as file:
        pickle._dump(inputTree, file)

def grabTree(filename):
    """从文件中读取决策树"""
    import pickle
    # open(filename,'r') 报如下错误：
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
    # 解决办法:open(filename,'rb')
    with open(filename,'rb') as file:
        return pickle._load(file)