import pandas as pd
import requests
import sys


def getRoadStatus(id,lat_min,lng_min,lat_max,lng_max,ak):
    locStr = str(lat_min) + "," + str(lng_min) + ";" + str(lat_max) + "," + str(lng_max)
    url='https://api.map.baidu.com/traffic/v1/bound?ak='+ak+'&bounds='+locStr+'&coord_type_input=gcj02&coord_type_output=gcj02'
    res = requests.get(url)
    data = res.json()
    statusCode = data.get('status')
    if(statusCode==302):  #302状态码（天配额超限，限制访问）
        print(statusCode)
        sys.exit(0)  #自动退出程序
    if data.get('road_traffic') is not None:
        evaluation = data.get('evaluation')
        status = evaluation.get('status')
        roadNum=len(data.get('road_traffic'))  
        print('id:',id,'status:',status,'raodNum:',roadNum,'statusCode:',statusCode)
        d={'id':id,'lat_min':lat_min,'lng_min':lng_min,'lat_max':lat_max,'lng_max':lng_max,'status':status,'roadNum':roadNum,'statusCode':statusCode}
        df = pd.DataFrame([d])
        df.to_csv('./百度地图_路况.csv',mode='a',index=False,header=False)
    else:
        print('id:',id,'无数据','statusCode:',statusCode)
        d={'id':id,'lat_min':lat_min,'lng_min':lng_min,'lat_max':lat_max,'lng_max':lng_max,'status':'?','roadNum':'?','statusCode':statusCode}
        df = pd.DataFrame([d])
        df.to_csv('./百度地图_路况.csv',mode='a',index=False,header=False)    

if __name__ == '__main__':
    # with open('./百度地图_路况.csv', 'a+',encoding='utf-8') as f:
    #     f.write("id,lat_min,lng_min,lat_max,lng_max,status,roadNum\n", )
    #     f.close()

    # 从csv等文件读取
    inputData=pd.read_csv('单元格经纬度.csv',encoding='utf-8')
    # 单元格的左下和右上经纬度（格式为：纬度,经度;纬度,经度）
    id=inputData.iloc[:,0]
    lat_min=inputData.iloc[:,1]
    lng_min=inputData.iloc[:,2]
    lat_max=inputData.iloc[:,3]
    lng_max=inputData.iloc[:,4]

    AkDict = {
    1: 'QFx12O7FPRlKTwMusa6cyWTC7gPGtaHc',
    2: 'C2UhQKyqptmCDgpk1NRs3eALUFotcfKM',
    3: 'Cm1l293GX2Em2nVagQUzBa7G7xKpKOzo',
    4: '9jBIrsKeC5duzToDmjBGOIoGYDB8Rg8U',
}                            
    AkNum = 2                    #由于一个ak只有2000次调用额度，这里加了一个计数器和一个key来在2000次之后换ak
    # for i in range(9055,35974):
    # for i in range(23148,35974):  #一共35975个,第一个数是要开始爬的id-1(或者上一个结束的id),结尾是35974
        # if(i%2000==0):
        #     AkNum+=1
    i=23148
    getRoadStatus(id[i],lat_min[i],lng_min[i],lat_max[i],lng_max[i],AkDict[AkNum])
    print('路况获取完成，已生成结果')