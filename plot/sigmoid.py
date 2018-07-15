from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt

def logistic(x):
    return 1/(1+np.exp(-x))

def plotLogistic():
    myfont = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    xValues = np.linspace(-5, 5, 10000)
    yValues = [logistic(x) for x in xValues]
    plt.plot(xValues, yValues)
    plt.scatter(0,0.5,c='red')
    plt.title("对数几率函数（logistic function）",fontproperties=myfont,fontsize=12)
    plt.xlabel("x",fontsize=12)
    plt.ylabel('logistic(x)',fontsize=12)
    plt.show()

def plotOverallLogistic():
    myfont = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    #xValues = np.append(np.arange(0,60,0.01),np.arange(-60,0,0.01))
    xValues = np.linspace(-60, 60, 120000)
    yValues = [logistic(x) for x in xValues]
    plt.plot(xValues,yValues)
    plt.scatter(0,0.5,c='red')
    plt.title("对数几率函数（logistic function）",fontproperties=myfont,fontsize=12)
    plt.xlabel("x",fontsize=12)
    plt.ylabel('logistic(x)',fontsize=12)
    plt.show()    
plotLogistic()
plotOverallLogistic()