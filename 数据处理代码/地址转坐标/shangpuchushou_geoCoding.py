# 获取广州各个商铺出售的经纬度

from urllib import parse
from urllib.request import urlopen
import pandas as pd
import requests
import re
import pandas as pd

def getAddressCode(biaoti,xingzhengqu,dibiao,jutiweizhi,dizhi,mianji,zongjia,yuan_mi):
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
        d={'标题':biaoti,'行政区':xingzhengqu,'地标':dibiao,'具体位置':jutiweizhi,'地址':dizhi,'建筑面积/平方米':mianji,'总价(万)':zongjia,'每平方米价格(元)':yuan_mi,
       '等级':level,'百度地图经度':lng, '百度地图纬度':lat,'GCJ02经度':x,'GCJ02纬度':y}
        data = pd.DataFrame(data=d)
        data.to_csv('./广州市商铺出售、经纬度.csv',mode='a',index=False,header=False,encoding='gbk')
        return data
    except Exception:
        d={'标题':biaoti,'行政区':xingzhengqu,'地标':dibiao,'具体位置':jutiweizhi,'地址':dizhi,'建筑面积/平方米':mianji,'总价(万)':zongjia,'每平方米价格(元)':yuan_mi,
       '等级':'','百度地图经度':'', '百度地图纬度':'','GCJ02经度':'','GCJ02纬度':''}
        data = pd.DataFrame(data=d)
        data.to_csv('./广州市商铺出售、经纬度.csv',mode='a',index=False,header=False,encoding='gbk')
        print("error!!!!!!!!!!!!!!!!!!!!")
if __name__ == '__main__':
    #从csv等文件读取，必须手动另存为csv，不能直接改后缀
    inputData=pd.read_csv('./58同城-商铺出售-全广州的数据.csv',encoding="gbk")

    biaoti=inputData.iloc[:,0]
    xingzhengqu=inputData.iloc[:,1]
    dibiao=inputData.iloc[:,2]
    jutiweizhi=inputData.iloc[:,3]
    dizhi=inputData.iloc[:,4]
    mianji=inputData.iloc[:,5]
    zongjia=inputData.iloc[:,6]
    yuan_mi=inputData.iloc[:,7]

    # with open('./广州市商铺出售、经纬度.csv', 'a+',encoding='gbk') as f:
    #     f.write("标题,行政区,地标,具体位置,地址,建筑面积/平方米,总价(万),每平方米价格(元),等级,百度地图经度,百度地图纬度,GCJ02经度,GCJ02纬度\n", )
    #     f.close()
    for i in range(551,824):  #csv中要开始的位置-2,表最后一列-1
    # for i in range(len(biaoti)):  
        getAddressCode(biaoti[i],xingzhengqu[i],dibiao[i],jutiweizhi[i],dizhi[i],mianji[i],zongjia[i],yuan_mi[i])

    print('商铺地址加载完成，已生成结果')