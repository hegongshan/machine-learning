from numpy import array, zeros, tile, shape
import operator
from os import listdir
from fileinput import filename

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#k-Nearest Neighbors
def classify0(inX,dataSet,labels,k):
    """k-近邻算法"""
    # 读取n*1的数据集第一维的长度n
    dataSetSize = dataSet.shape[0]
    # 将inX扩展为一个n*1的矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # 矩阵元素翻倍
    sqDiffMat = diffMat**2
    # 矩阵的每一行相加
    sqDistance = sqDiffMat.sum(axis=1)
    # 开根号，求得欧氏距离
    distances = sqDistance**0.5
    # 按数组值从小到大排序并返回其索引值
    soredDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[soredDistIndicies[i]]
        # get(key,default=None)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items()
                               ,key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 读取数据
def file2matrix(filename):
    with open(filename) as file:
        arrayOfLines = file.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
 
# 将数值归一化        
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))    
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals                           
    
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 前10%作为测试数据
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with : %d,the real anwser is: %d"%(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is : %f"%(errorCount/float(numTestVecs))) 
    
def img2vector(filename):
    returnVector = zeros((1,1024))
    with open(filename) as file:
        for i in range(32):
            lineStr = file.readline()
            for j in range(32):
                returnVector[0,32*i+j] = int(lineStr[j])
                
    return returnVector

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 获取文件名
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/%s"%fileNameStr)
    testFileList = listdir("testDigits")
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split("_")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("testDigits/%s"%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器预测结果：%d,实际的分类：%d"%(classifierResult,classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print("预测错误数量：%d"%errorCount)
    print("错误率：%f"%float(errorCount/mTest))
    
    