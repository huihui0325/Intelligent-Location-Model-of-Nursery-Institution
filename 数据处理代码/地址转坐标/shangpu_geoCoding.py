# 获取广州各个商铺出租的经纬度

from urllib import parse
from urllib.request import urlopen
import pandas as pd
import requests
import re
import pandas as pd

def getAddressCode(diqu,biaoti,xingzhengqu,dibiao,jutiweizhi,dizhi,mianji,wan_yue,yuan_mi_tian):
    try:
        url = "http://api.map.baidu.com/geocoding/v3/?address="+str(dizhi)+"&city=广东省广州市&output=json&ak=C2UhQKyqptmCDgpk1NRs3eALUFotcfKM"
  
        response = requests.get(url)
        text = response.text
        # print(type(text))
        pattern_lng = "lng\":(.*?),"
        pattern_lat = "lat\":(.*?)}"
        pattern_level = "level\":(.*?)}"
        lng = re.findall(pattern_lng, text)
        lat = re.findall(pattern_lat, text)
        level = re.findall(pattern_level, text)
        print("lng:" + lng[0])
        print("lat:" + lat[0])
        print("level:"+level[0])
        print("国测局（GCJ02）坐标")
        url1 = "http://api.map.baidu.com/geoconv/v1/?coords=" + lng[0] + "," + lat[0] + "&from=3&to=5&ak=C2UhQKyqptmCDgpk1NRs3eALUFotcfKM"
        text2 = requests.get(url1).text
        pattern_x = "x\":(.*?),"
        pattern_y = "y\":(.*?)}"
        x = re.findall(pattern_x, text2)
        y = re.findall(pattern_y, text2)
        print("x:" + x[0])
        print("y:" + y[0])
        d={'标题':biaoti,'行政区':diqu,'地标':dibiao,'具体位置':jutiweizhi,'地址':dizhi,'建筑面积':mianji,'万/月':wan_yue,'元每平方米每天':yuan_mi_tian,
       '等级':level,'百度地图经度':lng, '百度地图纬度':lat,'GCJ02经度':x,'GCJ02纬度':y}
        data = pd.DataFrame(data=d)
        data.to_csv('./商铺经纬度/'+diqu+'区商铺出租、经纬度.csv',mode='a',index=False,header=False,encoding='gbk')
        return data
    except Exception:
        d={'标题':biaoti,'行政区':diqu,'地标':dibiao,'具体位置':jutiweizhi,'地址':dizhi,'建筑面积':mianji,'万/月':wan_yue,'元每平方米每天':yuan_mi_tian,
       '等级':'','百度地图经度':'', '百度地图纬度':'','GCJ02经度':'','GCJ02纬度':''}
        data = pd.DataFrame(data=d)
        data.to_csv('./商铺经纬度/'+diqu+'区商铺出租、经纬度.csv',mode='a',index=False,header=False,encoding='gbk')
        print("error!!!!!!!!!!!!!!!!!!!!")
if __name__ == '__main__':
    #加载地区名，这里放在tuple里
    # places = ('岭东路95号保利高尔夫郡')
    #从csv等文件读取
    diqu='番禺' #黄埔 从化 越秀 南沙 荔湾 花都 增城 海珠 天河 白云 番禺
    # 必须手动另存为csv，不能直接改后缀
    inputData=pd.read_csv('./商铺出租文件/58同城-'+diqu+'-商铺出租.csv',encoding="gbk")

    biaoti=inputData.iloc[:,0]
    xingzhengqu=inputData.iloc[:,1]
    dibiao=inputData.iloc[:,2]
    jutiweizhi=inputData.iloc[:,3]
    dizhi=inputData.iloc[:,4]
    mianji=inputData.iloc[:,5]
    wan_yue=inputData.iloc[:,6]
    yuan_mi_tian=inputData.iloc[:,7]

    # with open('./商铺经纬度/'+diqu+'区商铺出租、经纬度.csv', 'a+',encoding='gbk') as f:
    #     f.write("标题,行政区,地标,具体位置,地址,建筑面积,万/月,元每平方米每天,等级,百度地图经度,百度地图纬度,GCJ02经度,GCJ02纬度\n", )
    #     f.close()
    for i in range(1348,1975):  #csv中要开始的位置-2,表最后一列-1
    # for i in range(len(biaoti)):  
        getAddressCode(diqu,biaoti[i],xingzhengqu[i],dibiao[i],jutiweizhi[i],dizhi[i],mianji[i],wan_yue[i],yuan_mi_tian[i])
    # getAddressCode(places)
    print('商铺地址加载完成，已生成结果')