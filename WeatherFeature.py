from tool import *
import pandas as pd
from math import isnan
import numpy as np

# 读取天气数据
columns_name1 = ['city_name', 'date', 'Max_tem', 'Min_tem', 'weather', 'wind_direction', 'wind_power']
weather_data = pd.read_table('/users/t-mac/desktop/data_group/weather_all.csv', sep=',', names=columns_name1)
weather_group = dict(list(weather_data.groupby(['city_name', 'date'])))
# 读取商家信息
columns_name2 = ['shop_id', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt', 'shop_level', 'cate_1_name',
                 'cate_2_name', 'cate_3_name']
shop_info = pd.read_table('/users/t-mac/desktop/data_group/dataset/shop_info.txt', sep=',', names=columns_name2)

shop_info.index = range(1, 2001)

user_pay_count = pd.read_csv('/users/t-mac/desktop/data_group/dataset/shop_pay_count.csv')

for i in range(0, len(user_pay_count.index)):
    user_pay_count.loc[i, 'date'] = trans_day(user_pay_count.loc[i, 'date'])

user_pay_count.index = user_pay_count['date']

good_weather = ['晴', '晴~多云', '多云~晴', '多云', '阴', '多云~阴', '阴~多云', '晴~阴', '阴~晴']

day_gaps = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -21, -30]
before_ = 'before_'
for id in range(1, 2001):
    #
    user_pay_id_count = pd.DataFrame(index=user_pay_count['date'],
                                     columns=['count', 'before_1', 'before_2', 'before_3', 'before_4', 'before_5',
                                              'before_6', 'before_7', 'before_8', 'before_9', 'before_10', 'before_11',
                                              'before_12', 'before_13', 'before_14', 'mean_before_1', 'mean_before_2',
                                              'mean_before_3', 'mean_before_4', 'mean_before_5',
                                              'mean_before_6', 'mean_before_7', 'mean_before_8', 'mean_before_9',
                                              'mean_before_10', 'mean_before_11',
                                              'mean_before_12', 'mean_before_13', 'mean_before_14', 'mean_before_21',
                                              'mean_before_30', 'min_before_1',
                                              'min_before_2', 'min_before_3', 'min_before_4', 'min_before_5',
                                              'min_before_6', 'min_before_7', 'min_before_8', 'min_before_9',
                                              'min_before_10', 'min_before_11',
                                              'min_before_12', 'min_before_13', 'min_before_14', 'min_before_21',
                                              'min_before_30', 'std_before_1',
                                              'std_before_2',
                                              'std_before_3', 'std_before_4', 'std_before_5',
                                              'std_before_6', 'std_before_7', 'std_before_8', 'std_before_9',
                                              'std_before_10', 'std_before_11',
                                              'std_before_12', 'std_before_13', 'std_before_14', 'std_before_21',
                                              'std_before_30', 'Mon', 'Tues',
                                              'Wed',
                                              'Thur',
                                              'Fri', 'Sat', 'Sun', 'good', 'general', 'bad', 'High_tem',
                                              'comfortable_tem', 'Low_tem'])
    file = '/users/t-mac/desktop/data_group/NewFeature_Weather/shop_feature_'
    city = shop_info.ix[id, 'city_name']
    for date in user_pay_id_count.index:

        # 处理天气特征
        # 处理温度
        date_otherFormat = trans_day_2(date)
        if weather_group[city, str(date_otherFormat)]['Max_tem'].values[0] > 30 and \
                        weather_group[city, str(date_otherFormat)]['Min_tem'].values[0] >= 22:
            user_pay_id_count.loc[date, 'High_tem'] = 1
        elif weather_group[city, date_otherFormat]['Max_tem'].values[0] <= 0 and \
                        weather_group[city, date_otherFormat]['Min_tem'].values[0] < -8:
            user_pay_id_count.loc[date, 'Low_tem'] = 1
        else:
            user_pay_id_count.loc[date, 'comfortable_tem'] = 1

        # 处理天气
        if weather_group[city, date_otherFormat]['weather'].values[0] in good_weather:
            user_pay_id_count.loc[date, 'good'] = 1
        elif str(weather_group[city, date_otherFormat]['weather'].values[0]).__contains__('暴|大|中|雷'):
            user_pay_id_count.loc[date, 'bad'] = 1
        else:
            user_pay_id_count.loc[date, 'general'] = 1

        # 处理count
        user_pay_id_count.loc[date, 'count'] = user_pay_count.loc[date, str(id)]

        # 处理星期特征
        day = datetime.datetime.strptime(date, '%Y/%m/%d')
        day = int(day.weekday() + 1)
        if day == 1:
            user_pay_id_count.loc[date, 'Mon'] = 1
        if day == 2:
            user_pay_id_count.loc[date, 'Tues'] = 1
        if day == 3:
            user_pay_id_count.loc[date, 'Wed'] = 1
        if day == 4:
            user_pay_id_count.loc[date, 'Thur'] = 1
        if day == 5:
            user_pay_id_count.loc[date, 'Fri'] = 1
        if day == 6:
            user_pay_id_count.loc[date, 'Sat'] = 1
        if day == 7:
            user_pay_id_count.loc[date, 'Sun'] = 1

        # 处理前14天count值
        j = 14
        for before_date_14 in date_range(move_day(date, -14), date)[:-1]:
            if before_date_14 not in user_pay_id_count.index:
                pass
            else:
                user_pay_id_count.loc[date, before_ + str(j)] = user_pay_count.ix[before_date_14, str(id)]
            j = j - 1
        # 处理滑动窗口特征：
        for gap in day_gaps:

            everyday_count = []
            for date2 in date_range(move_day(date, gap), date)[:-1]:

                if date2 not in user_pay_id_count.index:
                    pass
                else:
                    if user_pay_count.ix[date2, str(id)] != 0:
                        everyday_count.append(user_pay_count.ix[date2, str(id)])

            if len(date_range(move_day(date, gap), date)) == 2:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_1'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_1'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_1'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 3:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_2'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_2'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_2'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 4:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_3'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_3'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_3'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 5:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_4'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_4'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_4'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 6:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_5'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_5'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_5'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 7:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_6'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_6'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_6'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 8:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_7'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_7'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_7'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 9:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_8'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_8'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_8'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 10:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_9'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_9'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_9'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 11:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_10'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_10'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_10'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 12:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_11'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_11'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_11'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 13:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_12'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_12'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_12'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 14:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_13'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_13'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_13'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 15:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_14'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_14'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_14'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 22:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_21'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_21'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_21'] = int(np.array(everyday_count).std())

            if len(date_range(move_day(date, gap), date)) == 31:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'mean_before_30'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'min_before_30'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'std_before_30'] = int(np.array(everyday_count).std())
    # 删除0的日期
    for date in user_pay_id_count.index:
        if user_pay_id_count.loc[date, 'count'] == 0:
            user_pay_id_count = user_pay_id_count.drop([date])
    user_pay_id_count.fillna(0, inplace=True)
    file = file + str(id) + '.csv'
    user_pay_id_count.to_csv(file)
    print('第', id, '完成')

print(weather_data)
