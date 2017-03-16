import pandas as pd

import datetime
import numpy as np
from math import isnan
from tool import *

user_pay_count = pd.read_csv('/users/t-mac/desktop/data_group/dataset/20170112/shop_pay_count.csv')

for i in range(0, len(user_pay_count.index)):
    user_pay_count.loc[i, 'date'] = trans_day(user_pay_count.loc[i, 'date'])

user_pay_count.index = user_pay_count['date']


day_gaps = [-1, -2, -3, -4, -7, -14, -21, -30]
date_1 = '2016/11/03'
# print(move_day('2016/11/1', -1))
# print(date_range('2016/10/30','2016/11/01'))
for id in range(1, 2001):
    user_pay_id_count = pd.DataFrame(index=user_pay_count['date'],
                                     columns=['count', 'V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                                              'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean',
                                              'V9_day2_min',
                                              'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min',
                                              'V14_day21_min',
                                              'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std',
                                              'V19_day7_std',
                                              'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed',
                                              'Thur',
                                              'Fri', 'Sat', 'Sun'])


    user_pay_count.index = user_pay_count['date']
    for date in user_pay_id_count.index:
        # 处理count
        user_pay_id_count.loc[date, 'count'] = user_pay_count.loc[date, str(id)]

    user_date_20161101 = pd.DataFrame(index=['2016/11/03'],
                                      columns=['count', 'V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                                               'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean',
                                               'V9_day2_min',
                                               'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min',
                                               'V14_day21_min',
                                               'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std',
                                               'V19_day7_std',
                                               'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed',
                                               'Thur',
                                               'Fri', 'Sat', 'Sun'])

    # 处理星期特征
    day = datetime.datetime.strptime(date_1, '%Y/%m/%d')
    day = int(day.weekday() + 1)
    if day == 1:
        user_date_20161101.loc[date_1, 'Mon'] = 1
    if day == 2:
        user_date_20161101.loc[date_1, 'Tues'] = 1
    if day == 3:
        user_date_20161101.loc[date_1, 'Wed'] = 1
    if day == 4:
        user_date_20161101.loc[date_1, 'Thur'] = 1
    if day == 5:
        user_date_20161101.loc[date_1, 'Fri'] = 1
    if day == 6:
        user_date_20161101.loc[date_1, 'Sat'] = 1
    if day == 7:
        user_date_20161101.loc[date_1, 'Sun'] = 1
    # 处理滑动窗口特征：
    for gap in day_gaps:
        everyday_count = []

        for date2 in date_range(move_day(date_1, gap), date_1)[:-1]:

            if date2 not in user_pay_id_count.index:
                pass
            else:
                if user_pay_count.ix[date2, str(id)] != 0:
                    everyday_count.append(user_pay_count.ix[date2, str(id)])
        if len(date_range(move_day(date_1, gap), date_1)) == 2:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V1'] = str(np.array(everyday_count).mean())
        if len(date_range(move_day(date_1, gap), date_1)) == 3:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V2_day2_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V9_day2_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V16_day2_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date_1, gap), date_1)) == 4:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V3_day3_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V10_day3_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V17_day3_std'] = int(np.array(everyday_count).std())

        if len(date_range(move_day(date_1, gap), date_1)) == 5:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V4_day4_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V11_day4_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V18_day4_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date_1, gap), date_1)) == 8:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V5_day7_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V12_day7_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V19_day7_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date_1, gap), date_1)) == 15:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V6_day14_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V13_day14_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V20_day14_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date_1, gap), date_1)) == 22:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V7_day21_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V14_day21_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V21_day21_std'] = int(np.array(everyday_count).std())

        if len(date_range(move_day(date_1, gap), date_1)) == 31:
            if isnan(np.array(everyday_count).mean()) == False:
                user_date_20161101.loc[date_1, 'V8_day30_mean'] = int(np.array(everyday_count).mean())
                user_date_20161101.loc[date_1, 'V15_day30_min'] = int(np.array(everyday_count).min())
                user_date_20161101.loc[date_1, 'V22_day30_std'] = int(np.array(everyday_count).std())
    file = '/users/t-mac/desktop/data_group/20161103rmf/shop_feature_test_'
    file = file + str(id) + '.csv'
    user_date_20161101.fillna(0, inplace=True)
    user_date_20161101.to_csv(file)
    print('第', id, '完成')
print('finish')
