import matplotlib.pyplot as plt

decisionNode = dict(boxStyle="sawtooth",fc="0.8")
leafNode = dict(boxStyle="round4",fc="0.8")
arrow_args = dict(arrowStyle="<-")

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords="axes fraction",
                            xytext=centerPt,textcoords="axes fraction",
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    #plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    #plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    
def getNumLeafs(myTree):
    """获取叶子的数量"""
    numLeafs = 0
    # TypeError: 'dict_keys' object does not support indexing
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果某个键的值为字典，则递归执行
        if type(secondDict[key]).__name__=="dict":
            numLeafs += getNumLeafs(secondDict[key]) 
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """获取树的层次"""
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    """测试数据"""
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

def plotMidText(centerPt,parentPt,txtString):
    xMid = (parentPt[0]-centerPt[0])/2.0 + centerPt[0]
    yMid = (parentPt[1]-centerPt[1])/2.0 + centerPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    centerPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yoff)
    plotMidText(centerPt,parentPt,nodeTxt)
    plotNode(firstStr,centerPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],centerPt,str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),centerPt,leafNode)
            
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

    