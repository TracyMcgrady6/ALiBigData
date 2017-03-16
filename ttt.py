import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from math import isnan
from tool import *
from numpy import *

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

predict_data = pd.DataFrame(index=range(1, 2001), columns=['day1'])

date_in_range = date_range('2016/11/01', '2016/11/14')

error_big=[]
k = 1
for pre_date in date_in_range:

    user = pd.DataFrame(user_pay_count, index=[pre_date])

    user_pay_count = user_pay_count.append([user])
    user_pay_count.loc[pre_date, 'date'] = pre_date
    for id in range(1, 2001):
        # 特征化
        user_pay_id_count = pd.DataFrame(index=user_pay_count['date'],
                                         columns=['count', 'before_1', 'before_2', 'before_3', 'before_4', 'before_5',
                                                  'before_6', 'before_7', 'before_8', 'before_9', 'before_10',
                                                  'before_11',
                                                  'before_12', 'before_13', 'before_14', 'mean_before_1',
                                                  'mean_before_2',
                                                  'mean_before_3', 'mean_before_4', 'mean_before_5',
                                                  'mean_before_6', 'mean_before_7', 'mean_before_8', 'mean_before_9',
                                                  'mean_before_10', 'mean_before_11',
                                                  'mean_before_12', 'mean_before_13', 'mean_before_14',
                                                  'mean_before_21',
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

        user_pay_count.index = user_pay_count['date']

        for date in user_pay_id_count.index:
            # 处理count
            if isnan(user_pay_count.loc[date, str(id)]) == False:
                user_pay_id_count.loc[date, 'count'] = user_pay_count.loc[date, str(id)]
            else:
                pass

        user_date_20161101 = pd.DataFrame(index=[pre_date],
                                          columns=['count', 'before_1', 'before_2', 'before_3', 'before_4', 'before_5',
                                                   'before_6', 'before_7', 'before_8', 'before_9', 'before_10',
                                                   'before_11',
                                                   'before_12', 'before_13', 'before_14', 'mean_before_1',
                                                   'mean_before_2',
                                                   'mean_before_3', 'mean_before_4', 'mean_before_5',
                                                   'mean_before_6', 'mean_before_7', 'mean_before_8', 'mean_before_9',
                                                   'mean_before_10', 'mean_before_11',
                                                   'mean_before_12', 'mean_before_13', 'mean_before_14',
                                                   'mean_before_21',
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

        # file = '/users/t-mac/desktop/data_group/new_test_feature'+pre_date+'/shop_feature_'

        # 处理天气特征
        # 处理温度
        city = shop_info.ix[id, 'city_name']

        date_otherFormat = trans_day_2(pre_date)
        if weather_group[city, str(date_otherFormat)]['Max_tem'].values[0] > 30 and \
                        weather_group[city, str(date_otherFormat)]['Min_tem'].values[0] >= 22:
            user_date_20161101.loc[pre_date, 'High_tem'] = 1
        elif weather_group[city, date_otherFormat]['Max_tem'].values[0] <= 0 and \
                        weather_group[city, date_otherFormat]['Min_tem'].values[0] < -8:
            user_date_20161101.loc[pre_date, 'Low_tem'] = 1
        else:
            user_date_20161101.loc[pre_date, 'comfortable_tem'] = 1

        # 处理天气
        if weather_group[city, date_otherFormat]['weather'].values[0] in good_weather:
            user_date_20161101.loc[pre_date, 'good'] = 1
        elif str(weather_group[city, date_otherFormat]['weather'].values[0]).__contains__('暴|大|中|雷'):
            user_date_20161101.loc[pre_date, 'bad'] = 1
        else:
            user_date_20161101.loc[pre_date, 'general'] = 1

        # 处理星期特征
        day = datetime.datetime.strptime(pre_date, '%Y/%m/%d')
        day = int(day.weekday() + 1)
        if day == 1:
            user_date_20161101.loc[pre_date, 'Mon'] = 1
        if day == 2:
            user_date_20161101.loc[pre_date, 'Tues'] = 1
        if day == 3:
            user_date_20161101.loc[pre_date, 'Wed'] = 1
        if day == 4:
            user_date_20161101.loc[pre_date, 'Thur'] = 1
        if day == 5:
            user_date_20161101.loc[pre_date, 'Fri'] = 1
        if day == 6:
            user_date_20161101.loc[pre_date, 'Sat'] = 1
        if day == 7:
            user_date_20161101.loc[pre_date, 'Sun'] = 1
        # 处理前14天count值
        j = 14
        for before_date_14 in date_range(move_day(pre_date, -14), pre_date)[:-1]:
            if before_date_14 not in user_pay_id_count.index:
                pass
            else:
                user_date_20161101.loc[pre_date, before_ + str(j)] = user_pay_count.ix[before_date_14, str(id)]
            j = j - 1
        # 处理滑动窗口特征：
        for gap in day_gaps:

            everyday_count = []
            for date2 in date_range(move_day(pre_date, gap), pre_date)[:-1]:

                if date2 not in user_pay_id_count.index:
                    pass
                else:
                    if user_pay_count.ix[date2, str(id)] != 0:
                        everyday_count.append(user_pay_count.ix[date2, str(id)])

            if len(date_range(move_day(pre_date, gap), pre_date)) == 2:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_1'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_1'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_1'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 3:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_2'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_2'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_2'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 4:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_3'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_3'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_3'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 5:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_4'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_4'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_4'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 6:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_5'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_5'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_5'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 7:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_6'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_6'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_6'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 8:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_7'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_7'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_7'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 9:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_8'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_8'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_8'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 10:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_9'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_9'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_9'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 11:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_10'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_10'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_10'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 12:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_11'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_11'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_11'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 13:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_12'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_12'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_12'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 14:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_13'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_13'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_13'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 15:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_14'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_14'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_14'] = int(np.array(everyday_count).std())
            if len(date_range(move_day(pre_date, gap), pre_date)) == 22:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_21'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_21'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_21'] = int(np.array(everyday_count).std())

            if len(date_range(move_day(pre_date, gap), pre_date)) == 31:
                if isnan(np.array(everyday_count).mean()) == False:
                    user_date_20161101.loc[pre_date, 'mean_before_30'] = int(np.array(everyday_count).mean())
                    user_date_20161101.loc[pre_date, 'min_before_30'] = int(np.array(everyday_count).min())
                    user_date_20161101.loc[pre_date, 'std_before_30'] = int(np.array(everyday_count).std())

        file = '/users/t-mac/desktop/data_group/predict_test' + str(k) + '/shop_feature_test_'
        file = file + str(id) + '.csv'
        user_date_20161101.fillna(0, inplace=True)
        user_date_20161101.to_csv(file)
        print('第', id, '特征化完成')

        #############
        # 训练模型，预测值
        file_train = '/users/t-mac/desktop/data_group/NewFeature_Weather/shop_feature_' + str(id) + '.csv'
        file_test = '/users/t-mac/desktop/data_group/predict_test' + str(k) + '/shop_feature_test_' + str(id) + '.csv'

        feature_train = pd.read_csv(file_train)
        feature_test = pd.read_csv(file_test)

        X = feature_train[['before_1', 'before_2', 'before_3', 'before_4', 'before_5',
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
                           'comfortable_tem', 'Low_tem']]
        Y = feature_train['count']
        # 分割测试集
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

        X_predict = feature_test[['before_1', 'before_2', 'before_3', 'before_4', 'before_5',
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
                                  'comfortable_tem', 'Low_tem']]
        error = []
        predict = []

        vec = DictVectorizer(sparse=False)
        X_train = vec.fit_transform(X_train.to_dict(orient='record'))
        X_test = vec.fit_transform(X_test.to_dict(orient='record'))
        X_predict = vec.fit_transform(X_predict.to_dict(orient='record'))
        #
        # # 随机森林1
        # model1 = RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.2, max_leaf_nodes=100)
        # model1.fit(X_train, y_train)
        # model1_y_predict = model1.predict(X_test)
        # model1_predict = model1.predict(X_predict)
        #
        # error.append(mean_absolute_error(y_test, model1_y_predict))
        # predict.append(int(model1_predict))
        # print("rfm1 error ", mean_absolute_error(y_test, model1_y_predict), ",预测值为", model1_predict)

        # 随机森林2
        model2 = ExtraTreesRegressor(n_estimators=1000, max_depth=12, max_features=0.3, max_leaf_nodes=400)
        model2.fit(X_train, y_train)
        model2_y_predict = model2.predict(X_test)
        model2_predict = model2.predict(X_predict)

        error.append(mean_absolute_error(y_test, model2_y_predict))
        predict.append(int(model2_predict))
        print("rfm2 error ", mean_absolute_error(y_test, model2_y_predict), ",预测值为", model2_predict)
        #
        # XGBoost1
        # model3 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
        #                       colsample_bylevel=0.7)
        # model3.fit(X_train, y_train)
        # model3_y_predict = model3.predict(X_test)
        # model3_predict = model3.predict(X_predict)
        #
        # error.append(mean_absolute_error(y_test, model3_y_predict))
        # predict.append(int(model3_predict))
        # print("xgboost1 error ", mean_absolute_error(y_test, model3_y_predict), ",预测值为", model3_predict)
        #
        # # XGBoost2
        # model4 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
        #                       colsample_bylevel=0.7)
        # model4.fit(X_train, y_train)
        # model4_y_predict = model4.predict(X_test)
        # model4_predict = model4.predict(X_predict)
        #
        # error.append(mean_absolute_error(y_test, model4_y_predict))
        # predict.append(int(model4_predict))
        # print("xgboost2 error ", mean_absolute_error(y_test, model4_y_predict), ",预测值为", model4_predict)

        # XGBoost3
        model5 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                              colsample_bylevel=0.7, seed=10000)
        model5.fit(X_train, y_train)
        model5_y_predict = model5.predict(X_test)
        model5_predict = model5.predict(X_predict)

        error.append(mean_absolute_error(y_test, model5_y_predict))
        predict.append(int(model5_predict))
        print("xgboost3 error ", mean_absolute_error(y_test, model5_y_predict), ",预测值为", model5_predict)

        b = np.array(error)
        if b.min() > 30:
            error_big.append(id)
            # #1误差过大，取客流量平均值
            # mean_count = np.array(user_pay_count[str(id)])
            # mean_count = np.nan_to_num(mean_count)
            # print('平均绝对误差过大:', b.min(), ',取平均客流量：', int(mean_count[nonzero(mean_count)].mean()))
            # predict_data.loc[id, 'day1'] = int(mean_count[nonzero(mean_count)].mean())

            #2误差过大，取一周前客流量
            day_7_before=move_day(pre_date,-7)
            if user_pay_count.loc[day_7_before,str(id)]!=0:

                print('平均绝对误差过大1:', b.min(), ',取一周前客流量：', user_pay_count.loc[day_7_before,str(id)])
                predict_data.loc[id, 'day1']=user_pay_count.loc[day_7_before,str(id)]
                user_pay_count.loc[pre_date, str(id)] = user_pay_count.loc[day_7_before,str(id)]

            else:#取平均客流量
                mean_count = np.array(user_pay_count[str(id)])
                mean_count = np.nan_to_num(mean_count)
                print('平均绝对误差过大2:', b.min(), ',取平均客流量：', int(mean_count[nonzero(mean_count)].mean()))
                predict_data.loc[id, 'day1'] = int(mean_count[nonzero(mean_count)].mean())
                user_pay_count.loc[pre_date, str(id)] = int(mean_count[nonzero(mean_count)].mean())



        else:
            print('最好的预测值:', predict[np.where(b == b.min())[0][0]])
            predict_data.loc[id, 'day1'] = predict[np.where(b == b.min())[0][0]]
            user_pay_count.loc[pre_date, str(id)] = predict[np.where(b == b.min())[0][0]]

        # 将预测日期的客流量写入
        print(id, '预测完成!')

    predict_data.to_csv('/users/t-mac/desktop/data_group/predict/predict_data_' + str(k) + '.csv')
    user_pay_count.to_csv('/users/t-mac/desktop/data_group/dataset/shop_pay/shop_pay_' + str(k) + '.csv')
    k=k+1

er = pd.DataFrame(list(set(error_big)))
er.to_csv('/users/t-mac/desktop/error_id.csv')
print('finish')
