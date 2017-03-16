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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,ExtraTreesRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.feature_extraction import DictVectorizer

file_train = '/users/t-mac/desktop/data_group/NewFeature_Weather/shop_feature_2.csv'
feature_train = pd.read_csv(file_train)

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
print("rfm error ", mean_absolute_error(y_test, rfc_y_predict))

model4 = RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.2, max_leaf_nodes=100)
model4.fit(X_train, y_train)
model4_y_predict = model4.predict(X_test)
print("rfm2 error ", mean_absolute_error(y_test, model4_y_predict))

model17 = ExtraTreesRegressor(n_estimators=1000, max_depth=12, max_features=0.3, max_leaf_nodes=400)
model17.fit(X_train, y_train)
model17_y_predict = model17.predict(X_test)
print("etr error ", mean_absolute_error(y_test, model17_y_predict))


model1 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                      colsample_bylevel=0.7)
model1.fit(X_train, y_train)
model1_y_predict = model1.predict(X_test)
print("xgboost1 error ", mean_absolute_error(y_test, model1_y_predict))

model2 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
                      colsample_bylevel=0.7)
model2.fit(X_train, y_train)
model2_y_predict = model2.predict(X_test)
print("xgboost2 error ", mean_absolute_error(y_test, model2_y_predict))

model6 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7, seed=10000)
model6.fit(X_train, y_train)
model6_y_predict = model6.predict(X_test)
print("xgboost3 error ", mean_absolute_error(y_test, model6_y_predict))


xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
xgbc_y_predict=xgbc.predict(X_test)
print("mean absoliate error ",mean_absolute_error(y_test,xgbc_y_predict))


#支持向量基慢
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
svm_y_predict = linear_svr.predict(X_test)
print("svm error ", mean_absolute_error(y_test, svm_y_predict))

model3 = LinearSVR(tol=1e-7)
model3.fit(X_train, y_train)
model3_y_predict = model3.predict(X_test)
print("LinearSVR error ", mean_absolute_error(y_test, model3_y_predict))

lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)
print("linear error ", mean_absolute_error(y_test, lr_y_predict))
#
# gbc = GradientBoostingClassifier()
# gbc.fit(X_train, y_train)
# gbc_y_predict = gbc.predict(X_test)
# print("gbc error ", mean_absolute_error(y_test, lr_y_predict))
