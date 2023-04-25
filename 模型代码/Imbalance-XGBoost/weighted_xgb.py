# pip install imbalance-xgboost
import numpy as np #导入NumPy数学工具箱
import pandas as pd #导入Pandas数据处理工具箱
from sklearn.model_selection import train_test_split #拆分数据集
from sklearn.preprocessing import MinMaxScaler # 导入缩放器
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report # 导入分类报告
from sklearn.metrics import confusion_matrix # 导入混淆矩阵
import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
import csv

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

# 定义一个函数显示混淆矩阵
def show_matrix(y_test, y_pred): 
    cm = confusion_matrix(y_test,y_pred) # 调用混淆矩阵
    plt.title("Weighted-XGBoost Confusion Matrix") # 标题
    sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False) # 热力图设定
    plt.show() # 显示混淆矩阵

def show_report(y_test, y_pred): 
    if y_test.shape != (2000,1):
        y_test = y_test.values # 把Panda series转换成Numpy array
        y_test = y_test.reshape((len(y_test),1)) # 转换成与y_pred相同的形状 
    print(classification_report(y_test,y_pred,labels=[0, 1])) #调用分类报告  

if __name__ == '__main__':
    df = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple).csv')
    y = df ['tuoyujigou']
    X = df.drop(['id','tuoyujigou','gongyuan','yiyuan','shangchang'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # xgboster_weight = imb_xgb(special_objective='weighted')
    # CV_weight_booster = GridSearchCV(xgboster_weight, {"imbalance_alpha":[2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1]}) #"imbalance_alpha":[0,0.5,1.0,1.5,2.0,2.5,3.0]
    # CV_weight_booster.fit(X_train, y_train)
    # opt_weight_booster = CV_weight_booster.best_estimator_
    # print("最佳weight参数组合:",CV_weight_booster.best_params_)
    # y_pred=opt_weight_booster.predict_determine(X_test)

    # base = imb_xgb(special_objective='weighted', imbalance_alpha=2.3)
    # base.fit(X_train, y_train)
    # print('base fit over')
    # y_pred = base.predict_determine(X_test)
    # show_matrix(y_test, y_pred)
    # show_report(y_test, y_pred)

    base = imb_xgb(special_objective='weighted', imbalance_alpha=2.3)
    history=base.fit(X_train, y_train)
    print('base fit over')
    # y_predAll = base.predict_determine(X)
    # show_matrix(y, y_predAll)
    # show_report(y, y_predAll)
    # yy=base.predict_sigmoid(X)
    # y_csv=pd.DataFrame({'pred':yy})
    # y_csv.to_csv('./所有点的概率.csv',mode='a',index=False,header=True)
    df1 = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple_N).csv')
    y1 = df1 ['tuoyujigou']
    X1 = df1.drop(['id','tuoyujigou','gongyuan','yiyuan','shangchang'], axis=1)
    y_predN = base.predict_determine(X1)
    yy=base.predict_sigmoid(X1)
    show_matrix(y1,y_predN)
    # print(yy)
    # print(X1.shape)
    # raw_output = opt_weight_booster.predict(X_test, y=None) 
    # sigmoid_output = opt_weight_booster.predict_sigmoid(X_test, y=None) 
    # prob_output = opt_weight_booster.predict_two_class(X_test, y=None) 
    # print(sigmoid_output)

    # df2 = pd.read_csv('./data4(simple_N).csv')  #输入带位置的坐标
    # y_csv=pd.DataFrame({'id':df2['id'],'bd_lat':df2['bd_lat'],'bd_lng':df2['bd_lng'],'pred':yy})
    # # print(y_predRes.shape)
    # with open('D:\托育机构选址\选址数据源\代码管理\可视化\备选点概率.csv', 'a') as f:
    #     f.write("id,bd_lat,bd_lng,pred,\n",)
    #     f.close()
    # y_csv.to_csv('D:\托育机构选址\选址数据源\代码管理\可视化\备选点概率.csv',mode='a',index=False,header=True)