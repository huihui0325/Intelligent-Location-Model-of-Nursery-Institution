# ROC曲线、AUC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import pandas as pd #导入Pandas数据处理工具箱
from sklearn.model_selection import train_test_split #拆分数据集
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb

if __name__ == '__main__':
    df = pd.read_csv('D:\托育机构选址\模型\Easy Model\sample\data4(simple).csv')
    y = df ['tuoyujigou']
    X = df.drop(['id','tuoyujigou'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # model = imb_xgb(special_objective='weighted', imbalance_alpha=2.3)
    # model.fit(X_train, y_train)
    # print('model fit over')
    model = imb_xgb(special_objective='focal', focal_gamma=1.9)
    model.fit(X_train, y_train)
    print('model fit over')

    # 预测正例的概率
    y_pred_prob=model.predict_two_class(X_test)[:,1]
    # y_pred_prob ,返回两列，第一列代表类别0,第二列代表类别1的概率
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob, pos_label=1)
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
    plt.title('AUC of Focal-XGBoost')
    plt.show()

        # y_pred = model.predict_determine(X_test)
#     fpr_1, tpr_1, threshold_1 = roc_curve(y_test, y_pred)  # 计算FPR和TPR
#     auc_1 = auc(fpr_1, tpr_1)  # 计算AUC值

# # precision_1, recall_1, threshold_1 = precision_recall_curve(y_1, yp_1)  # 计算Precision和Recall
# # aupr_1 = auc(recall_1, precision_1)  # 计算AUPR值
# # precision_2, recall_2, threshold_2 = precision_recall_curve(y_2, yp_2)  # 计算Precision和Recall
# # aupr_2 = auc(recall_2, precision_2)  # 计算AUPR值
# # precision_3, recall_3, threshold_3 = precision_recall_curve(y_3, yp_3)  # 计算Precision和Recall
# # aupr_3 = auc(recall_3, precision_3)  # 计算AUPR值
# # precision_4, recall_4, threshold_4 = precision_recall_curve(y_4, yp_4)  # 计算Precision和Recall
# # aupr_4 = auc(recall_4, precision_4)  # 计算AUPR值


#     # 3.绘制曲线
#     line_width = 1  # 曲线的宽度
#     plt.figure(figsize=(8, 5))  # 图的大小
#     plt.plot(fpr_1, tpr_1, lw=line_width, label='imbalance-XGBoost (AUC = %0.4f)' % auc_1, color='blue')

# # plt.plot(precision_3, recall_3, lw=line_width, label='HSIC-MKL + LP-S (AUPR = %0.4f)' % aupr_3, color='red')
# # plt.plot(precision_1, recall_1, lw=line_width, label='HSIC-MKL + LP-P (AUPR = %0.4f)' % aupr_1, color='blue')
# # plt.plot(precision_4, recall_4, lw=line_width, label='Mean weighted + LP-S (AUPR = %0.4f)' % aupr_4, color='green')
# # plt.plot(precision_2, recall_2, lw=line_width, label='Mean weighted + LP-P (AUPR = %0.4f)' % aupr_2, color='orange')


#     # 4.坐标轴范围和标题
#     plt.xlim([0.0, 1.0])  # 限定x轴的范围
#     plt.ylim([0.0, 1.0])  # 限定y轴的范围
#     # plt.xticks(range(0, 10, 1)) # 修改x轴的刻度
#     # plt.yticks(range(0, 10, 1)) # 修改y轴的刻度

#     plt.xlabel('False Positive Rate')  # x坐标轴标题
#     plt.ylabel('True Positive Rate')  # y坐标轴标题
#     # plt.xlabel('Recall')
#     # plt.ylabel('Precision')

#     # plt.title('Receiver Operating Characteristic')  # 图标题

#     plt.grid()  # 在图中添加网格

#     plt.legend(loc="lower right")  # 显示图例并指定图例位置


#     # 5.中文处理问题
#     # plt.rcParams['font.sans-serif'] = ['SimHei']
#     # plt.rcParams['axes.unicode_minus'] = False


#     # 6.展示图片和保存
#     # plt.savefig('AUC.tif', dpi=300)
#     plt.show()