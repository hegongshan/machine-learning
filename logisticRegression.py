from numpy import mat, shape, ones, exp, array, arange, random

def loadDataSet():
    dataMatrix = [];labelMatrix = []
    with open('testSet.txt') as file:
        for line in file.readlines():
            lineArr = line.strip().split()
            dataMatrix.append([1.0,float(lineArr[0]),float(lineArr[1])])
            labelMatrix.append(int(lineArr[2]))
        return dataMatrix,labelMatrix

def sigmoid(inputVector):
    """sigmoid阶跃函数"""
    return 1.0/(1+exp(-inputVector))

def gradientAscent(dataMatrix,labelMatrix):
    """梯度上升算法"""
    dataMatrix = mat(dataMatrix)
    labelMatrix = mat(labelMatrix).transpose() # 转置
    n = shape(dataMatrix)[1]
    alpha = 0.001
    maxCycle = 500
    weights = ones((n,1))
    k = 0
    while k < maxCycle:
        h = sigmoid(dataMatrix*weights)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights

def randomGradientAscent(dataMatrix,labelMatrix):
    """随机梯度上升算法"""
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        print(str(dataMatrix[i])+"-----------------------------")
        print(dataMatrix[i]*weights)
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = labelMatrix[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def improvedRandomGradientAscent(dataMatrix,labelMatrix,numIter=150):
    """改进的随机梯度上升算法"""
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randomIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randomIndex]*weights))
            error = labelMatrix[randomIndex]-h
            weights = weights + alpha * error * dataMatrix[randomIndex]
            del dataIndex[randomIndex]
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMatrix,labelMatrix = loadDataSet()
    dataArr = array(dataMatrix)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def classifyVector(inX,weights):
    p = sigmoid(sum(inX*weights))
    if p > 0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest():
    trainingSet = [];trainingLabels = []
    with open('horseColicTraining.txt') as trainFile:
        for line in trainFile.readlines():
            currentLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currentLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currentLine[21]))
    trainWeights = randomGradientAscent(array(trainingSet),trainingLabels)
    errorCount = 0;numTestVec = 0.0
    with open('horseColicTest.txt') as testFile:
        for line in testFile.readlines():
            numTestVec += 1.0
            currentLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currentLine[i]))
            if int(classifyVector(array(lineArr), trainWeights)) != int(currentLine[21]):
                errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print("错误率%f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("10次迭代后，平均的错误率为%f" % (errorSum/float(numTests)))  
    
