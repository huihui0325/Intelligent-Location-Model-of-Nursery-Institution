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


# 定义一个函数显示混淆矩阵
def show_matrix(y_test, y_pred): 
    cm = confusion_matrix(y_test,y_pred) # 调用混淆矩阵
    plt.title("Focal-XGBoost Confusion Matrix") # 标题
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
    X = df.drop(['id','tuoyujigou'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # xgboster_focal = imb_xgb(special_objective='focal')
    # CV_focal_booster = GridSearchCV(xgboster_focal, {"focal_gamma":[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4]}) #"focal_gamma":[0,0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3.0]
    # CV_focal_booster.fit(X_train, y_train)
    # opt_focal_booster = CV_focal_booster.best_estimator_
    # print("最佳focal参数组合:",CV_focal_booster.best_params_)
    # y_pred=opt_focal_booster.predict_determine(X_test)
    # y_predAll=opt_focal_booster.predict_determine(X)
    base = imb_xgb(special_objective='focal', focal_gamma=1.9)
    base.fit(X_train, y_train)
    print('base fit over')
    y_pred = base.predict_determine(X_test)
    show_matrix(y_test, y_pred)
    show_report(y_test, y_pred)

    # base = imb_xgb(special_objective='focal', focal_gamma=1.9)
    # base.fit(X_train, y_train)
    # print('base fit over')
    # y_predAll = base.predict_determine(X)
    # show_matrix(y, y_predAll)
    # show_report(y, y_predAll)

    # raw_output = opt_focal_booster.predict(X_test, y=None) 
    # sigmoid_output = opt_focal_booster.predict_sigmoid(X_test, y=None) 
    # prob_output = opt_focal_booster.predict_two_class(X_test, y=None) 
    # print(sigmoid_output)