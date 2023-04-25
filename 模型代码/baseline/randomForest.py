import numpy as np #导入NumPy数学工具箱
import pandas as pd #导入Pandas数据处理工具箱
import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split #拆分数据集
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler # 导入特征缩放器
from keras.models import Sequential # 导入Keras序贯模型
from keras.layers import Dense # 导入Keras密集连接层
from sklearn.metrics import classification_report # 导入分类报告
from sklearn.metrics import confusion_matrix # 导入混淆矩阵
from tensorflow import keras
from sklearn.metrics import (f1_score,accuracy_score)
from sklearn.preprocessing import MinMaxScaler # 导入数据缩放器
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # 导入随机森林分类器
from sklearn.model_selection import StratifiedKFold # 导入K折验证工具
from sklearn.model_selection import GridSearchCV # 导入网格搜索工具
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

# 显示训练过程中的学习曲线
def show_history(history): 
    # 训练集和验证集的损失
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 训练集和验证集的准确率
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 定义一个函数显示分类报告
def show_report(y_test, y_pred): 
    if y_test.shape != (2000,1):
        y_test = y_test.values # 把Panda series转换成Numpy array
        y_test = y_test.reshape((len(y_test),1)) # 转换成与y_pred相同的形状 
    print(classification_report(y_test,y_pred,labels=[0, 1])) #调用分类报告  

# 定义一个函数显示混淆矩阵
def show_matrix(y_test, y_pred): 
    cm = confusion_matrix(y_test,y_pred) # 调用混淆矩阵
    plt.title("RandomForest Confusion Matrix") # 标题
    sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False) # 热力图设定
    plt.show() # 显示混淆矩阵



if __name__ == '__main__':
    df = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple).csv')
    y = df ['tuoyujigou']
    X = df.drop(['id','tuoyujigou'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
    rf.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = rf.predict(X_test)
    
    # show_matrix(y_test, y_pred)
    # show_report(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=1)
    #pos_label，代表真阳性标签，就是说是分类里面的好的标签，这个要看你的特征目标标签是0,1，还是1,2
    roc_auc = metrics.auc(fpr, tpr)  #auc为Roc曲线下的面积
    # print(roc_auc)
    plt.figure(figsize=(8,6))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, 'r',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('AUC of RandomForest')
    plt.show()

    # kfold = StratifiedKFold(n_splits=10) # 10折验证
    # rf = RandomForestClassifier() # 随机森林
    # # 对随机森林算法进行参数优化
    # rf_param_grid = {"max_depth": [None],
    #             "max_features": [3, 5, 12],
    #             "min_samples_split": [2, 5, 10],
    #             "min_samples_leaf": [3, 5, 10],
    #             "bootstrap": [False],
    #             "n_estimators" :[100,300],
    #             "criterion": ["gini"]}
    # rf_gs = GridSearchCV(rf,param_grid = rf_param_grid, cv=kfold, 
    #                     scoring="accuracy", n_jobs= 10, verbose = 1)
    # rf_gs.fit(X_resampled, y_resampled) # 用优化后的参数拟合训练数据集
    # y_hat_rfgs = rf_gs.predict(X_test) # 用随机森林算法的最佳参数进行预测
    # print("参数优化后随机森林测试准确率:", accuracy_score(y_test.T, y_hat_rfgs))
    # cm_rfgs = confusion_matrix(y_test,y_hat_rfgs) # 显示混淆矩阵
    # plt.figure(figsize=(4,4))
    # plt.title("Random Forest (Best Score) Confusion Matrix")
    # sns.heatmap(cm_rfgs,annot=True,cmap="Blues",fmt="d",cbar=False)
    # print("最佳参数组合:",rf_gs.best_params_)














