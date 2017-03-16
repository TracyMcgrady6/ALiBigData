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
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.feature_extraction import DictVectorizer


predict_data=pd.DataFrame(index=range(1,2001),columns=['day1'])

for id in range(1, 2001):
    file_train = '/users/t-mac/desktop/data_group/feature_shop/shop_feature_' + str(id) + '.csv'
    file_test = '/users/t-mac/desktop/data_group/20161103rmf/shop_feature_test_' + str(id) + '.csv'

    feature_train = pd.read_csv(file_train)
    feature_test = pd.read_csv(file_test)

    X = feature_train[['V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                             'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean', 'V9_day2_min',
                             'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min', 'V14_day21_min',
                             'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std', 'V19_day7_std',
                             'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed', 'Thur',
                             'Fri', 'Sat', 'Sun']]
    Y = feature_train['count']

    X_test = feature_test[['V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                           'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean', 'V9_day2_min',
                           'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min', 'V14_day21_min',
                           'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std', 'V19_day7_std',
                           'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed', 'Thur',
                           'Fri', 'Sat', 'Sun']]

    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_y_predict = rfc.predict(X_test)

    # xgbc=XGBClassifier()
    # xgbc.fit(X_train,y_train)
    # xgbc_y_predict=xgbc.predict(X_test)

    print(rfc_y_predict[0])
    predict_data.loc[id,'day1'] = int(rfc_y_predict[0])
    print(id, '预测完成')
predict_data.to_csv('/users/t-mac/desktop/data_group/predict/predict_data_by_rmf_1103.csv')

print('finish')
