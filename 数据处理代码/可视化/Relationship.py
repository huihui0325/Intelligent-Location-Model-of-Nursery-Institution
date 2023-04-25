# 导入数据可视化所需要的库
import numpy as np #导入NumPy数学工具箱
import pandas as pd #导入Pandas数据处理工具箱
import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
from pylab import *

if __name__ == '__main__':
    df = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(zhongwen).csv',encoding='utf-8')
    # df.head() #读入数据并显示前面几行的内容，确保已经成功的读入数据
    df=df.drop(['id','托育机构'],axis=1)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    # #对所有的标签和特征两两显示其相关性的热力图(heatmap)
    sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
    plt.show() #plt代表英文plot,就是画图的意思
