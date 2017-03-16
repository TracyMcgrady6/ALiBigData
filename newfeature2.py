import pandas as pd
import datetime
import numpy as np
from math import isnan
from tool import *

# user_pay = pd.read_csv('/users/t-mac/desktop/data_group/dataset/shop_pay_count.csv')
Holiday = {'2016/01/01', '2016/01/02', '2016/01/03', '2016/02/06', '2016/02/07', '2016/02/08',
           '2016/02/09', '2016/02/10', '2016/02/11', '2016/02/12', '2016/02/13', '2016/04/02',
           '2016/04/03', '2016/04/04', '2016/04/30', '2016/05/01', '2016/05/02', '2016/06/09', '2016/06/10',
           '2016/06/11', '2016/09/15', '2016/09/16', '2016/09/17', '2016/10/01', '2016/10/01', '2016/10/02',
           '2016/10/03', '2016/10/04', '2016/10/05', '2016/10/06', '2016/10/07'}

user_pay_count = pd.read_csv('/users/t-mac/desktop/data_group/dataset/shop_pay_count.csv')

# user_pay_id_1 = pd.read_csv('/users/t-mac/desktop/data_group/newfeature/shop_id_1_feature.csv')

for i in range(0, len(user_pay_count.index)):
    user_pay_count.loc[i, 'date'] = trans_day(user_pay_count.loc[i, 'date'])

user_pay_count.index = user_pay_count['date']

print(user_pay_count.head())
day_gaps = [-1, -2, -3, -4, -7, -14, -21, -30]
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
    file = '/users/t-mac/desktop/data_group/feature_shop/shop_feature_'
    for date in user_pay_id_count.index:

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
                    user_pay_id_count.loc[date, 'V1'] = str(np.array(everyday_count).mean())
            if len(date_range(move_day(date, gap), date)) == 3:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V2_day2_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V9_day2_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V16_day2_std'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 4:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V3_day3_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V10_day3_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V17_day3_std'] = int(np.array(everyday_count).std())

            if len(date_range(move_day(date, gap), date)) == 5:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V4_day4_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V11_day4_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V18_day4_std'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 8:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V5_day7_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V12_day7_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V19_day7_std'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 15:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V6_day14_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V13_day14_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V20_day14_std'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(date, gap), date)) == 22:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V7_day21_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V14_day21_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V21_day21_std'] = int(np.array(everyday_count).std())

            if len(date_range(move_day(date, gap), date)) == 31:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_pay_id_count.loc[date, 'V8_day30_mean'] = int(np.array(everyday_count).mean())
                    user_pay_id_count.loc[date, 'V15_day30_min'] = int(np.array(everyday_count).min())
                    user_pay_id_count.loc[date, 'V22_day30_std'] = int(np.array(everyday_count).std())
    # 删除0的日期
    for date in user_pay_id_count.index:
        if user_pay_id_count.loc[date, 'count'] == 0:
            user_pay_id_count = user_pay_id_count.drop([date])
    user_pay_id_count.fillna(0, inplace=True)
    file = file + str(id) + '.csv'
    user_pay_id_count.to_csv(file)
    print('第', id, '完成')

# user_pay_id_1.to_csv('/users/t-mac/desktop/data_group/newfeature/shop_id_1_feature_1102.csv')
print('finish')
