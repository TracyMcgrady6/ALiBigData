import pandas as pd
import datetime
# 转化日期格式
def trans_day(day_str, time_regex='%Y/%m/%d'):
    t = datetime.datetime.strptime(day_str, time_regex)
    day_str = datetime.datetime.strftime(t, time_regex)
    return day_str


user_pay_id = pd.read_csv('/users/t-mac/desktop/data_group/dataset/shop_pay_count.csv')
for i in range(0, len(user_pay_id.index)):
    user_pay_id.loc[i, 'date'] =trans_day(user_pay_id.loc[i, 'date'])

user_pay_id.index = user_pay_id['date']
user_pay_id = pd.DataFrame(user_pay_id,
                             columns=['date', 'count', 'shop', 'V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                                      'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean', 'V9_day2_min',
                                      'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min', 'V14_day21_min',
                                      'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std', 'V19_day7_std',
                                      'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed', 'Thur',
                                      'Fri', 'Sat', 'Sun'])
user_pay_id = user_pay_id.drop(['date'], axis=1)
print(user_pay_id)
for i in range(1,2001):
    for date in user_pay_id.index:
        user_pay_id.loc[date,'shop']=i
    pass
print(user_pay_id)