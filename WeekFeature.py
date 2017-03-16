#encoding:utf-8
import pandas as pd
from datetime import datetime,date



columns_name=['date','Max_tem','Min_tem','weather','wind_direction','wind_power']
weather_data = pd.read_table('/users/t-mac/desktop/data_group/city_weather/hangzhou',sep=',',names=columns_name)

user_view = pd.read_csv('/users/t-mac/desktop/data_group/dataset/user_pay.txt')
user_view.time_stamp = user_view.time_stamp.str[:10]

customer_flow = user_view.groupby(['shop_id', 'time_stamp']).size()

print(len(customer_flow[23]))
length=len(customer_flow[23])
user_view_week = pd.DataFrame(index=range(0,length),columns=['time_stamp','week','Mon','Tues','Wed','Thur','Fri','Sat','Sun'
                                                             ,'sundy','other','High_tem','comfortable','Low_tem','count_pay'])
for i in range(0,length):
    user_view_week.loc[i,'time_stamp']= customer_flow[23].index[i]
    day = datetime.strptime(customer_flow[23].index[i], '%Y-%m-%d')
    user_view_week.loc[i, 'week'] =int(day.weekday()+1)
    day = int(day.weekday()+1)
    if day==1:
        user_view_week.loc[i, 'Mon'] = 1
    if day==2:
        user_view_week.loc[i, 'Tues'] = 1
    if day==3:
        user_view_week.loc[i, 'Wed'] = 1
    if day==4:
        user_view_week.loc[i, 'Thur'] = 1
    if day==5:
        user_view_week.loc[i, 'Fri'] = 1
    if day==6:
        user_view_week.loc[i, 'Sat'] = 1
    if day==7:
        user_view_week.loc[i, 'Sun'] = 1
    #下面处理天气特征：
    for j in range(0, len(weather_data.index)):
        if customer_flow[23].index[i]==weather_data['date'].values[j]:
            if int(weather_data['Max_tem'].values[j]) > 30:
                user_view_week.loc[i, 'High_tem'] = 1
            elif int(weather_data['Max_tem'].values[j]) < 0:
                user_view_week.loc[i, 'Low_tem'] = 1
            else :
                user_view_week.loc[i, 'comfortable'] = 1
            if weather_data['weather'].values[j]=='晴' or weather_data['weather'].values[j]=='多云' or weather_data['weather'].values[j]=='多云~晴' or weather_data['weather'].values[j]=='晴~多云':
                user_view_week.loc[i, 'sundy'] = 1
            else:
                user_view_week.loc[i, 'other'] = 1
    #添加客流量
    user_view_week.loc[i, 'count_pay'] =customer_flow[23].values[i]

user_view_week.fillna(0,inplace=True)

user_view_week.to_csv('/users/t-mac/desktop/data_group/myfeature/feature_id_23.csv')
print(user_view_week)
