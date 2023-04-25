# 百度地图，获取百度坐标
# 百度地图坐标是在GCJ-02坐标的基础上进行二次加密偏移形成的新坐标

from urllib import parse
from urllib.request import urlopen
import pandas as pd
import requests
import re
import pandas as pd

def getAddressCode(place,address):
    try:
        url = "http://api.map.baidu.com/geocoding/v3/?address="+address+"&city=广东省广州市&output=json&ak=QFx12O7FPRlKTwMusa6cyWTC7gPGtaHc"
        response = requests.get(url)
        text = response.text
        print(type(text))
        pattern_lng = "lng\":(.*?),"
        pattern_lat = "lat\":(.*?)}"
        lng = re.findall(pattern_lng, text)
        lat = re.findall(pattern_lat, text)
        print("lng:" + lng[0])
        print("lat:" + lat[0])
        print("国测局（GCJ02）坐标")
        url1 = "http://api.map.baidu.com/geoconv/v1/?coords=" + lng[0] + "," + lat[0] + "&from=3&to=5&ak=QFx12O7FPRlKTwMusa6cyWTC7gPGtaHc"
        text2 = requests.get(url1).text
        pattern_x = "x\":(.*?),"
        pattern_y = "y\":(.*?)}"
        x = re.findall(pattern_x, text2)
        y = re.findall(pattern_y, text2)
        print("x:" + x[0])
        print("y:" + y[0])
        # data = pd.DataFrame(data = [[address,lng,lat,x,y]],columns=['地址','百度地图经度','百度地图纬度','GCJ02经度','GCJ02纬度'])
        d = {'托育店铺':place,'地址': address, '百度地图经度':lng, '百度地图纬度':lat,'GCJ02经度':x,'GCJ02纬度':y}
        data = pd.DataFrame(data=d)
        data.to_csv('./百度地图-美团托育.csv',mode='a',index=False,header=False)
        return data
    except Exception:
        print("error")
if __name__ == '__main__':
    #加载地区名，这里放在tuple里
    # places = ('新塘镇解放北路113号新星未来教育城2楼','宦溪西路兰亭盛荟四季里商业街1号143-144号')
    #从csv等文件读取
    inputData=pd.read_csv('../../托育机构/美团托育店铺.csv',encoding="gbk")
    places=inputData.iloc[:,0]
    address=inputData.iloc[:,1]

    with open('./goeinfo.csv', 'a+',encoding='utf-8') as f:
        f.write("托育店铺,地址,百度地图经度,百度地图纬度,GCJ02经度,GCJ02纬度\n", )
        f.close()
    for i in range(len(places)):
        getAddressCode(places[i],address[i])
    # getAddressCode(places)
    print('地区加载完成，已生成结果')