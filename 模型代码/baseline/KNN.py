# 测试集正样本占总正样本10%，且正负比例模拟原来的。训练集和验证集只有1400多个样本

import numpy as np #导入NumPy数学工具箱
import pandas as pd #导入Pandas数据处理工具箱
import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split #拆分数据集
from sklearn.preprocessing import StandardScaler # 导入特征缩放器
from keras.models import Sequential # 导入Keras序贯模型
from keras.layers import Dense # 导入Keras密集连接层
from sklearn.metrics import classification_report # 导入分类报告
from sklearn.metrics import confusion_matrix # 导入混淆矩阵
from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler # 导入数据缩放器
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier # 导入kNN算法
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

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
    plt.title("KNN Confusion Matrix") # 标题
    sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False) # 热力图设定
    plt.show() # 显示混淆矩阵


if __name__ == '__main__':
    df = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple).csv')
    y = df ['tuoyujigou']
    X = df.drop(['id','tuoyujigou'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # 进行特征缩放
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    k = 5 # 设定初始K值为5
    kNN = KNeighborsClassifier(n_neighbors = k)  # kNN模型
    kNN.fit(X_train, y_train) # fit,就相当于是梯度下降
    y_pred = kNN.predict(X_test) # 预测测试集的标签
    # print(y_pred)
    # y_pred = np.round(y_pred) # 四舍五入，将分类概率值转换成0/1整数值

    # show_report(y_test, y_pred)
    # show_matrix(y_test, y_pred)
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
    plt.title('AUC of KNN')
    plt.show()

    # 寻找最佳K值
    # f1_score_list = []
    # acc_score_list = []
    # for i in range(1,15):
    #     kNN = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    #     kNN.fit(X_resampled, y_resampled)
    #     acc_score_list.append(kNN.score(X_test, y_test))
    #     y_pred = kNN.predict(X_test) # 预测心脏病结果
    #     f1_score_list.append(f1_score(y_test, y_pred))
    # index = np.arange(1,15,1)
    # plt.plot(index,acc_score_list,c='blue',linestyle='solid')
    # plt.plot(index,f1_score_list,c='red',linestyle='dashed')
    # plt.legend(["Accuracy", "F1 Score"])
    # plt.xlabel("K value")
    # plt.ylabel("Score")
    # plt.grid('false')
    # plt.show()
    # kNN_acc = max(f1_score_list)*100
    # print("Maximum kNN Score is {:.2f}%".format(kNN_acc))

