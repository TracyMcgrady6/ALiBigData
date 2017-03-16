import pandas as pd
import tools
import datetime
import numpy as np
from math import isnan

# user_pay = pd.read_csv('/users/t-mac/desktop/data_group/dataset/shop_pay_count.csv')
Holiday = {'2016/01/01', '2016/01/02', '2016/01/03', '2016/02/06', '2016/02/07', '2016/02/08',
           '2016/02/09', '2016/02/10', '2016/02/11', '2016/02/12', '2016/02/13', '2016/04/02',
           '2016/04/03', '2016/04/04', '2016/04/30', '2016/05/01', '2016/05/02', '2016/06/09', '2016/06/10',
           '2016/06/11', '2016/09/15', '2016/09/16', '2016/09/17', '2016/10/01', '2016/10/01', '2016/10/02',
           '2016/10/03', '2016/10/04', '2016/10/05', '2016/10/06', '2016/10/07'}


def move_day(day_str, offset, time_regex='%Y/%m/%d'):
    '''
        计算day_str偏移offset天后的日期
    :param
        day_str: str 原时间
        offset: str 要偏移的天数
        time_regex: str 时间字符串的正则式
    :return:
        day_str: str 运算之后的结果时间, 同样以time_regex的格式返回
    --------
        如 move_day('20151228', 1)返回 '20151229'
    '''
    day = datetime.datetime.strptime(day_str, time_regex).date()
    day = day + datetime.timedelta(days=offset)
    day_str = datetime.datetime.strftime(day, time_regex)
    return day_str


# 转化日期格式
def trans_day(day_str, time_regex='%Y/%m/%d'):
    t = datetime.datetime.strptime(day_str, time_regex)
    day_str = datetime.datetime.strftime(t, time_regex)
    return day_str


def date_range(begin, end, time_regex='%Y/%m/%d'):
    '''
        生成begin到end的每一天的一个list
    :param
        begin: str 开始时间
        end: str 结束时间
        time_regex: str 时间格式的正则表达式
    :argument
        begin需要小于等于end
    :return:
        day_range: list
    --------
        如 date_range('20151220', '20151223')返回 ['20151220', '20151221', '20151222', '20151223']
    '''
    day_range = []
    day = datetime.datetime.strptime(begin, time_regex).date()
    while True:
        day_str = datetime.datetime.strftime(day, time_regex)
        day_range.append(day_str)
        if day_str == end:
            break
        day = day + datetime.timedelta(days=1)
    return day_range


user_pay_id_1 = pd.read_csv('/users/t-mac/desktop/data_group/newfeature/shop_id_1_feature.csv')

for i in range(0, len(user_pay_id_1.index)):
    user_pay_id_1.loc[i, 'date'] = trans_day(user_pay_id_1.loc[i, 'date'])

user_pay_id_1.index = user_pay_id_1['date']
user_pay_id_1 = pd.DataFrame(user_pay_id_1,
                             columns=['date', 'count', 'V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                                      'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean', 'V9_day2_min',
                                      'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min', 'V14_day21_min',
                                      'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std', 'V19_day7_std',
                                      'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed', 'Thur',
                                      'Fri', 'Sat', 'Sun'])
user_pay_id_1 = user_pay_id_1.drop(['date'], axis=1)
day_gaps = [-1, -2, -3, -4, -7, -14, -21, -30]

for date in user_pay_id_1.index:
    # 处理星期特征
    day = datetime.datetime.strptime(date, '%Y/%m/%d')
    day = int(day.weekday() + 1)
    if day == 1:
        user_pay_id_1.loc[date, 'Mon'] = 1
    if day == 2:
        user_pay_id_1.loc[date, 'Tues'] = 1
    if day == 3:
        user_pay_id_1.loc[date, 'Wed'] = 1
    if day == 4:
        user_pay_id_1.loc[date, 'Thur'] = 1
    if day == 5:
        user_pay_id_1.loc[date, 'Fri'] = 1
    if day == 6:
        user_pay_id_1.loc[date, 'Sat'] = 1
    if day == 7:
        user_pay_id_1.loc[date, 'Sun'] = 1

    # 处理滑动窗口特征：
    for gap in day_gaps:
        everyday_count = []
        for date2 in date_range(move_day(date, gap), date)[:-1]:

            if date2 not in user_pay_id_1.index:
                pass
            else:
                if user_pay_id_1.ix[date2, 'count'] != 0:
                    everyday_count.append(user_pay_id_1.ix[date2, 'count'])
        if len(date_range(move_day(date, gap), date)) == 2:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V1'] = str(np.array(everyday_count).mean())
        if len(date_range(move_day(date, gap), date)) == 3:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V2_day2_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V9_day2_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V16_day2_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date, gap), date)) == 4:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V3_day3_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V10_day3_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V17_day3_std'] = int(np.array(everyday_count).std())

        if len(date_range(move_day(date, gap), date)) == 5:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V4_day4_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V11_day4_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V18_day4_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date, gap), date)) == 8:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V5_day7_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V12_day7_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V19_day7_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date, gap), date)) == 15:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V6_day14_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V13_day14_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V20_day14_std'] = int(np.array(everyday_count).std())
        if len(date_range(move_day(date, gap), date)) == 22:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V7_day21_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V14_day21_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V21_day21_std'] = int(np.array(everyday_count).std())

        if len(date_range(move_day(date, gap), date)) == 31:
            if isnan(np.array(everyday_count).mean()) == False:
                user_pay_id_1.loc[date, 'V8_day30_mean'] = int(np.array(everyday_count).mean())
                user_pay_id_1.loc[date, 'V15_day30_min'] = int(np.array(everyday_count).min())
                user_pay_id_1.loc[date, 'V22_day30_std'] = int(np.array(everyday_count).std())

for date in user_pay_id_1.index:
    if user_pay_id_1.loc[date, 'count'] == 0:
        print('!!!!!')
        user_pay_id_1 = user_pay_id_1.drop([date])
user_pay_id_1.fillna(0, inplace=True)
user_pay_id_1.to_csv('/users/t-mac/desktop/data_group/newfeature/shop_id_1_feature_1102.csv')
print('finish')
