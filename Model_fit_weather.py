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

predict_data = pd.DataFrame(index=range(1, 2001), columns=['day1'])

for id in range(1, 2001):
    file_train = '/users/t-mac/desktop/data_group/NewFeature_Weather/shop_feature_' + str(id) + '.csv'
    file_test = '/users/t-mac/desktop/data_group/20161101newtest/shop_feature_test_' + str(id) + '.csv'

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

    # 随机森林1
    model1 = RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.2, max_leaf_nodes=100)
    model1.fit(X_train, y_train)
    model1_y_predict = model1.predict(X_test)
    model1_predict = model1.predict(X_predict)

    error.append(mean_absolute_error(y_test, model1_y_predict))
    predict.append(int(model1_predict))
    print("rfm1 error ", mean_absolute_error(y_test, model1_y_predict),",预测值为",model1_predict)

    # 随机森林2
    model2 = ExtraTreesRegressor(n_estimators=1000, max_depth=12, max_features=0.3, max_leaf_nodes=400)
    model2.fit(X_train, y_train)
    model2_y_predict = model2.predict(X_test)
    model2_predict = model2.predict(X_predict)

    error.append(mean_absolute_error(y_test, model2_y_predict))
    predict.append(int(model2_predict))
    print("rfm2 error ", mean_absolute_error(y_test, model2_y_predict),",预测值为",model2_predict)

    # XGBoost1
    model3 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                      colsample_bylevel=0.7)
    model3.fit(X_train, y_train)
    model3_y_predict = model3.predict(X_test)
    model3_predict = model3.predict(X_predict)

    error.append(mean_absolute_error(y_test, model3_y_predict))
    predict.append(int(model3_predict))
    print("xgboost1 error ", mean_absolute_error(y_test, model3_y_predict),",预测值为",model3_predict)

    # XGBoost2
    model4 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
                      colsample_bylevel=0.7)
    model4.fit(X_train, y_train)
    model4_y_predict = model4.predict(X_test)
    model4_predict = model4.predict(X_predict)

    error.append(mean_absolute_error(y_test, model4_y_predict))
    predict.append(int(model4_predict))
    print("xgboost2 error ", mean_absolute_error(y_test, model4_y_predict),",预测值为",model4_predict)

    # XGBoost3
    model5 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7, seed=10000)
    model5.fit(X_train, y_train)
    model5_y_predict = model5.predict(X_test)
    model5_predict = model5.predict(X_predict)

    error.append(mean_absolute_error(y_test, model5_y_predict))
    predict.append(int(model5_predict))
    print("xgboost3 error ", mean_absolute_error(y_test, model5_y_predict),",预测值为",model5_predict)

    b=np.array(error)

    print('最好的预测值是',predict[np.where(b==b.min())[0][0]])
    predict_data.loc[id, 'day1'] = predict[np.where(b==b.min())[0][0]]
    print(id, '预测完成!')
predict_data.to_csv('/users/t-mac/desktop/data_group/predict/predict_data_1101.csv')

print('finish')
