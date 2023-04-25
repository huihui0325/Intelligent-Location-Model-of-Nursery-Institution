import json
import pandas as pd
if __name__ == '__main__':
    # 读取CSV文件
    # csvData = pd.read_csv(r'./已有托育机构分布（美团）.csv', header = 0)  
    # csvData = pd.read_csv(r'./备选点概率.csv', header = 0)  
    csvData = pd.read_csv(r'./概率前20地点.csv', header = 0)  

    # 读取CSV文件包含的列名并转换为list
    columns = csvData.columns.tolist()
    # print(columns)

    # # 所有备选
    # for i in range(35151):
    #     # 创建空字典
    #     outPut = {}
    #     # 将CSV文件转为字典
    #     for col in columns:
    #         outPut[col] = str(csvData.loc[i, col]) # 这里一定要将数据类型转成字符串，否则会报错
    #     # 将字典转为json格式
    #     jsonData = json.dumps(outPut) # 注意此处是dumps，不是dump
    #     # 保存json文件
    #     with open(r'./全部备选点.json', 'a') as jsonFile:
    #         jsonFile.write(',') #到时候再删掉多的逗号，前后加上中括号
    #         jsonFile.write(jsonData)

    # 托育机构
    for i in range(20): #821
        # 创建空字典
        outPut = {}
        # 将CSV文件转为字典
        for col in columns:
            outPut[col] = str(csvData.loc[i, col]) # 这里一定要将数据类型转成字符串，否则会报错
        # 将字典转为json格式
        jsonData = json.dumps(outPut) # 注意此处是dumps，不是dump
        # 保存json文件
        with open(r'./概率前20地点.json', 'a') as jsonFile:
            jsonFile.write(',') #到时候再删掉多的逗号，加上中括号
            jsonFile.write(jsonData)
