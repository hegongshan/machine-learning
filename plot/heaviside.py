from matplotlib.font_manager import FontProperties

import matplotlib.pyplot as plt
import numpy as np

def heaviside(x):
    if x < 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        return 1

def plotHeaviside():
    myfont = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    xValues = np.append(np.arange(0,5,0.001),np.arange(-5,0,0.001))
    yValues = [heaviside(x) for x in xValues]
    plt.scatter(xValues, yValues,s=2)
    plt.title("heaviside阶跃函数",fontproperties=myfont,fontsize=12)
    plt.xlabel("x",fontsize=12)
    plt.ylabel('heaviside(x)',fontsize=12)
    plt.show()
    
plotHeaviside()