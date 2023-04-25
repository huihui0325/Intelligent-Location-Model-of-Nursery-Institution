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
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler # 导入数据缩放器
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # 导入随机森林分类器

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
    plt.title("ANN Confusion Matrix") # 标题
    sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False) # 热力图设定
    plt.show() # 显示混淆矩阵

if __name__ == '__main__':
    dfP = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple_P).csv') #正样本
    dfN = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple_N).csv') #负样本

    y_P= dfP['tuoyujigou']
    X_P = dfP.drop(['tuoyujigou','id'], axis=1)
    # print("样本比例:\n",y.value_counts()/len(y))  #0：0.97717  1：0.02823

    y_N= dfN['tuoyujigou']
    X_N = dfN.drop(['tuoyujigou','id'], axis=1)

    X_traVal_P, X_test_P, y_traVal_P, y_test_P = train_test_split(X_P, y_P, test_size=0.1, random_state=0) #取10%正样本作为测试集正样本()，剩余90%作为训练集和验证集正样本
    # print("X_traVal_P:",len(X_traVal_P)) # 83
    # print("X_test_P:",len(X_test_P)) # 738,需要3554个负样本
    X_traVal_N, X_test_N, y_traVal_N, y_test_N = train_test_split(X_N, y_N, test_size=0.1011, random_state=0) #取真实比例负样本作为测试集负样本，剩余作为训练集和验证集负样本
    # print("X_traVal_N:",len(X_traVal_N)) # 31597
    # print("X_test_N:",len(X_test_N)) # 3554

    X_test=pd.concat([X_test_N,X_test_P],axis=0) #横着堆叠
    y_test=pd.concat([y_test_N,y_test_P],axis=0) #横着堆叠
    # print("X_test:",len(X_test))  #一共3637个数据

    X_traVal=pd.concat([X_traVal_P,X_traVal_N],axis=0)
    y_traVal=pd.concat([y_traVal_P,y_traVal_N],axis=0)
    # print("X_traVal:",len(X_traVal))  #一共32335个数据

    # 随机欠采样
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_traVal, y_traVal)
    print("样本数据:",Counter(y_resampled))
    # X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled,  test_size=0.3, random_state=0)  #划分训练集和验证集

    # 进行特征缩放
    scaler = preprocessing.MinMaxScaler()
    X_resampled = scaler.fit_transform(X_resampled)
    X_test = scaler.transform(X_test)

    # 深层神经网络
    ann = Sequential() # 创建一个序贯ANN模型
    ann.add(Dense(units=12, input_dim=13, activation = 'relu')) # 添加输入层
    ann.add(Dense(units=24, activation = 'relu')) # 添加隐层
    # ann.add(Dropout(0.5)) # 添加Dropout
    ann.add(Dense(units=48, activation = 'relu')) # 添加隐层
    # ann.add(Dropout(0.5)) # 添加Dropout
    ann.add(Dense(units=96, activation = 'relu')) # 添加隐层
    # ann.add(Dropout(0.5)) # 添加Dropout
    ann.add(Dense(units=192, activation = 'relu')) # 添加隐层
    # ann.add(Dropout(0.5)) # 添加Dropout
    ann.add(Dense(units=1, activation = 'sigmoid')) # 添加输出层
    
    # 编译神经网络，指定优化器，损失函数，以及评估标准
    ann.compile(optimizer = 'adam',           #优化器，还可以试试RMSprop
                loss = 'binary_crossentropy', #损失函数  
                metrics = ['acc'])       #评估指标
    history = ann.fit(X_resampled, y_resampled, # 指定训练集
                  epochs=30,        # 指定训练的轮次
                  batch_size=64,    # 指定数据批量
                  validation_data=(X_test, y_test)) #指定验证集,这里为了简化模型，直接用测试集数据进行验证
    
    show_history(history) # 调用这个函数，并将神经网络训练历史数据作为参数输入
    
    y_pred = ann.predict(X_test,batch_size=10) # 预测测试集的标签
    # print(y_pred)
    y_pred = np.round(y_pred) # 四舍五入，将分类概率值转换成0/1整数值

    show_report(y_test, y_pred)
    show_matrix(y_test, y_pred)















