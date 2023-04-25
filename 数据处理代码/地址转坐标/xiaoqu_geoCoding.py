# 获取广州各个小区的经纬度

from urllib import parse
from urllib.request import urlopen
import pandas as pd
import requests
import re
import pandas as pd

def getAddressCode(xiaoqu,xingzhengqu,dibiao,dizhi,zaishou,zaizu,nianfen,fangjia,zhang,die):
    try:
        url = "http://api.map.baidu.com/geocoding/v3/?address="+str(dizhi)+xiaoqu+"&city=广东省广州市&output=json&ak=QFx12O7FPRlKTwMusa6cyWTC7gPGtaHc"
  
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
        url1 = "http://api.map.baidu.com/geoconv/v1/?coords=" + lng[0] + "," + lat[0] + "&from=3&to=5&ak=QFx12O7FPRlKTwMusa6cyWTC7gPGtaHc"
        text2 = requests.get(url1).text
        pattern_x = "x\":(.*?),"
        pattern_y = "y\":(.*?)}"
        x = re.findall(pattern_x, text2)
        y = re.findall(pattern_y, text2)
        print("x:" + x[0])
        print("y:" + y[0])
        d={'小区':xiaoqu,'行政区':xingzhengqu,'地标':dibiao,'地址':dizhi,'在售':zaishou,'在租':zaizu,'建成年份':nianfen,
        '平均房价':fangjia,'涨幅百分比':zhang,'跌幅百分比':die,'等级':level,'百度地图经度':lng, '百度地图纬度':lat,'GCJ02经度':x,'GCJ02纬度':y}
        data = pd.DataFrame(data=d)
        data.to_csv('./小区经纬度/'+xingzhengqu+'区房价、经纬度.csv',mode='a',index=False,header=False,encoding='gbk')
        return data
    except Exception:
        d={'小区':xiaoqu,'行政区':xingzhengqu,'地标':dibiao,'地址':dizhi,'在售':zaishou,'在租':zaizu,'建成年份':nianfen,
        '平均房价':fangjia,'涨幅百分比':zhang,'跌幅百分比':die,'等级':'','百度地图经度':'', '百度地图纬度':'','GCJ02经度':'','GCJ02纬度':''}
        data = pd.DataFrame(data=d)
        data.to_csv('./小区经纬度/'+xingzhengqu+'区房价、经纬度.csv',mode='a',index=False,header=False,encoding='gbk')
        print("error!!!!!!!!!!!!!!!!!!!!")
if __name__ == '__main__':
    #加载地区名，这里放在tuple里
    # places = ('岭东路95号保利高尔夫郡')
    #从csv等文件读取
    diqu='番禺'
    # 必须手动另存为csv，不能直接改后缀
    inputData=pd.read_csv('./小区文件/房天下-'+diqu+'房价.csv',encoding="gbk")

    xiaoqu=inputData.iloc[:,0]
    xingzhengqu=inputData.iloc[:,1]
    dibiao=inputData.iloc[:,2]
    dizhi=inputData.iloc[:,3]
    zaishou=inputData.iloc[:,4]
    zaizu=inputData.iloc[:,5]
    nianfen=inputData.iloc[:,6]
    fangjia=inputData.iloc[:,7]
    zhang=inputData.iloc[:,8]
    die=inputData.iloc[:,9]

    # with open('./小区经纬度/'+diqu+'区房价、经纬度.csv', 'a+',encoding='gbk') as f:
    #     f.write("小区,行政区,地标,地址,在售,在租,建成年份,平均房价,涨幅百分比,跌幅百分比,等级,百度地图经度,百度地图纬度,GCJ02经度,GCJ02纬度\n", )
    #     f.close()
    # for i in range(640,963):  #csv中要开始的位置-2,表最后一列-1
    for i in range(len(xiaoqu)):  
        getAddressCode(xiaoqu[i],xingzhengqu[i],dibiao[i],dizhi[i],zaishou[i],zaizu[i],nianfen[i],fangjia[i],zhang[i],die[i])
    # getAddressCode(places)
    print('地区加载完成，已生成结果')