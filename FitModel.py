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
from sklearn.feature_extraction import DictVectorizer

feature = pd.read_csv('/users/t-mac/desktop/data_group/newfeature/shop_id_1_feature_v4.csv')
feature_test = pd.read_csv('/users/t-mac/desktop/data_group/newfeature/test1102.csv')

X = feature[['V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
             'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean', 'V9_day2_min',
             'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min', 'V14_day21_min',
             'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std', 'V19_day7_std',
             'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed', 'Thur',
             'Fri', 'Sat', 'Sun']]
Y = feature['count']

X_pre = feature_test[['V1', 'V2_day2_mean', 'V3_day3_mean', 'V4_day4_mean',
                      'V5_day7_mean', 'V6_day14_mean', 'V7_day21_mean', 'V8_day30_mean', 'V9_day2_min',
                      'V10_day3_min', 'V11_day4_min', 'V12_day7_min', 'V13_day14_min', 'V14_day21_min',
                      'V15_day30_min', 'V16_day2_std', 'V17_day3_std', 'V18_day4_std', 'V19_day7_std',
                      'V20_day14_std', 'V21_day21_std', 'V22_day30_std', 'Mon', 'Tues', 'Wed', 'Thur',
                      'Fri', 'Sat', 'Sun']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

vec =DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
print("mean squrae error linear is",mean_squared_error(y_test,rfc_y_predict))
print("mean absoliate error ",mean_absolute_error(y_test,rfc_y_predict))

predict=rfc.predict(X_pre)
print(predict)

xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
xgbc_y_predict=xgbc.predict(X_test)

print("mean squrae error linear is",mean_squared_error(y_test,xgbc_y_predict))
print("mean absoliate error ",mean_absolute_error(y_test,xgbc_y_predict))

#
# ss_x=StandardScaler()
# ss_y=StandardScaler()
#
# X_train=ss_x.fit_transform(X_train)
# X_test=ss_x.fit_transform(X_test)
# y_train=ss_y.fit_transform(y_train)
# y_test=ss_y.fit_transform(y_test)
#
# lr =LinearRegression()
# lr.fit(X_train,y_train)
# lr_y_predict=lr.predict(X_test)
#
# predict=lr.predict(X_pre)
# print(predict)
# print("linearRegression is  ",lr.score(X_test,y_test))
# print("R-sq",r2_score(y_test,lr_y_predict))
# print("mean squrae error linear is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
# print("mean absoliate error ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
#
#
# sgdr = SGDRegressor()
# sgdr.fit(X_train,y_train)
# sgdr_y_predict = sgdr.predict(X_test)
#
# print("linearRegression is  ",sgdr.score(X_test,y_test))
# print("R-sq",r2_score(y_test,sgdr_y_predict))
# print("mean squrae error linear is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
# print("mean absoliate error ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
# predict=sgdr.predict(X_pre)
# print(predict)
