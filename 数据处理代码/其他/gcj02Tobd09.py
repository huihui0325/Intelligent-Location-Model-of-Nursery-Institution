import math
import csv
import pandas as pd

def gcj02_to_bd09(lon, lat):
    x_pi = math.pi * 3000.0 / 180.0
    z = math.sqrt(lon * lon + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lon) + 0.000003 * math.cos(lon * x_pi)
    bd_lon = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lon, bd_lat


if __name__ == '__main__':
    inputData=pd.read_csv('D:\托育机构选址\选址数据源\代码管理\坐标转换\data_gcj02.csv',encoding="gbk")    
    id=inputData.iloc[:,0]
    center_lat=inputData.iloc[:,5]
    center_lng=inputData.iloc[:,6]

    # with open('./data_bd09.csv', 'a+',encoding='gbk') as f:
    #     f.write("id,center_lat,center_lng,bd_lat,bd_lng\n")
    #     f.close()

    # bdj02_lat=pd.DataFrame()
    # bdj02_lng=pd.DataFrame()
    # for i in range(len(id)):
    #     bd_lon, bd_lat = gcj02_to_bd09(center_lng[i],center_lat[i])
    #     # d={'center_lat':lat,'center_lng':lon,'bd_lat':bd_lat,'bd_lng':bd_lon}
    #     # data = pd.DataFrame(data=d)
    #     # data.to_csv('./data_bd09.csv',mode='a',header=False,encoding='gbk')
    #     bdj02_lat=bdj02_lat.append(bd_lat)
    #     bdj02_lng=bdj02_lng.append(bd_lon)
    # data=pd.DataFrame({'id':i,'center_lat':center_lat,'center_lng':center_lng,'bd_lat':bdj02_lat,'bd_lng':bdj02_lng})
    # data.to_csv('./data_bd09.csv',index=True,header=False,encoding='gbk')  
    
    with open('D:\托育机构选址\选址数据源\代码管理\坐标转换\data_bd09.csv',"w",newline="") as file:
        writter = csv.writer(file)
        writter.writerow(["id","center_lat","center_lng","bd_lat","bd_lng\n"])
        for index,row in inputData.iterrows():  
            bd_lon, bd_lat = gcj02_to_bd09(row["center_lng"],row["center_lat"])
            writter.writerow([row["id"],row["center_lat"],row["center_lng"],bd_lat,bd_lon]) 
        file.close()
    print("坐标转换完毕")