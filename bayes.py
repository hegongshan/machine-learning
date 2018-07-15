"""
朴素贝叶斯分类器
"""
from numpy import *
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVector = [0,1,0,1,0,1]
    return postingList,classVector

def createVocabularyList(dataSet):
    """创建词汇列表"""
    vocabularySet = set([])
    for document in dataSet:
        vocabularySet = vocabularySet | set(document)
    return list(vocabularySet)

def setOfWords2Vector(vocabularyList,inputSet):
    # returnVector = []
    # for i in range(len(vocabularyList)):
    #     returnVector.append(0)
    returnVector = [0]*len(vocabularyList)
    for word in inputSet:
        if word in vocabularyList:
            returnVector[vocabularyList.index(word)] = 1
    return returnVector

def bagOfWords2Vector(vocabularyList,inputSet):
    """多项式模型（词袋模型）"""
    returnVector = [0]*len(vocabularyList)
    for word in inputSet:
        if word in vocabularyList:
            returnVector[vocabularyList.index(word)] += 1
    return returnVector

def textParse(bigString):
    import re
    listOfTokens = re.split('\\W+',bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]

def spamTest():
    docList = [];classList = [];fullText = []
    for i in range(1,26):
        with open('../email/spam/%d.txt' % i,encoding='iso-8859-1') as file:
            wordList = textParse(file.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
        with open('../email/ham/%d.txt' % i,encoding='iso-8859-1') as file:
            wordList = textParse(file.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
    vocabularyList = createVocabularyList(docList)
    trainingSet = list(range(50));testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMatrix = [];trainClasses = []
    for docIndex in trainingSet:
        trainMatrix.append(setOfWords2Vector(vocabularyList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNavieBayes(array(trainMatrix), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vector(vocabularyList,docList[docIndex])
        if classifyNavieBayes(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("The error rate is %f" %(float(errorCount)/len(testSet)))

def trainNavieBayes(trainMatrix,trainClassVector):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 计算侮辱性文档的概率
    pAbusive = sum(trainClassVector)/float(numTrainDocs)
    
    #p0Num = zeros(numWords); p1Num = zeros(numWords)
    #p0Denum = 0.0; p1Denum = 0.0
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denum = 2.0; p1Denum = 2.0
    for i in range(numTrainDocs):
        if(trainClassVector[i]==1):
            p1Num += trainMatrix[i]
            p1Denum += sum(trainMatrix[i])
            #print(str(p1Denum)+":"+str(trainMatrix[i])+"\n")
        else:
            p0Num += trainMatrix[i] 
            p0Denum += sum(trainMatrix[i])
            # print(str(p0Denum)+":"+str(trainMatrix[i])+"\n")
    #p1Vector = p1Num / p1Denum
    #p0Vector = p0Num / p0Denum
    p1Vector = log(p1Num / p1Denum)
    p0Vector = log(p0Num / p0Denum)
    return p0Vector,p1Vector,pAbusive

def classifyNavieBayes(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNavieBayes():
    listOfPosts,listClasses = loadDataSet()
    vocabList = createVocabularyList(listOfPosts)
    trainMatrix = []
    for postInDoc in listOfPosts:
        trainMatrix.append(setOfWords2Vector(vocabList, postInDoc))
    p0V,p1V,pAb = trainNavieBayes(array(trainMatrix), array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vector(vocabList, testEntry))
    print("%s classified as : %d"%(testEntry,classifyNavieBayes(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vector(vocabList, testEntry))
    print("%s classified as : %d"%(testEntry,classifyNavieBayes(thisDoc, p0V, p1V, pAb)))
    

def calcMostFrequency(vocabularyList,fullText):
    import operator
    freqDict = {}
    for token in vocabularyList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabularyList = createVocabularyList(docList)
    top30Words = calcMostFrequency(vocabularyList,fullText)
    for pairW in top30Words.keys():
        if pairW in vocabularyList:
            vocabularyList.remove(pairW)
    trainingSet = list(range(2*minLen));testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMatrix = [];trainClasses = []
    for docIndex in trainingSet:
        trainMatrix.append(bagOfWords2Vector(vocabularyList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNavieBayes(array(trainMatrix), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vector(vocabularyList, docList[docIndex])
        if classifyNavieBayes(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("The error rate is %f" % (float(errorCount)/len(testSet)))
    return vocabularyList,p0V,p1V