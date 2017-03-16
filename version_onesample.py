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
from sklearn.metrics import  r2_score,mean_squared_error,mean_absolute_error
from sklearn.feature_extraction import DictVectorizer

feature = pd.read_csv('/users/t-mac/desktop/data_group/myfeature/feature_id_23.csv')


X = feature[['Mon','Tues','Wed','Thur','Fri','Sat','Sun','sundy','other','High_tem','comfortable','Low_tem','before14','before13','before12','before11','before10','before9','before8',
             'before7','before6','before5','before4','before3','before2','before']]
Y = feature['count_pay']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=33)
# linreg = LinearRegression()
#
# linreg.fit(X_train, y_train)
# print(linreg.intercept_)#截距
# print(linreg.coef_)#系数


ss_x=StandardScaler()
ss_y=StandardScaler()

X_train=ss_x.fit_transform(X_train)
X_test=ss_x.fit_transform(X_test)
y_train=ss_y.fit_transform(y_train)
y_test=ss_y.fit_transform(y_test)

lr =LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

sgdr = SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict = sgdr.predict(X_test)

print("linearRegression is  ",lr.score(X_test,y_test))
print("R-sq",r2_score(y_test,lr_y_predict))
print("mean squrae error linear is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print("mean absoliate error ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
# print(y_pred2)


linear_svr=SVR(kernel='rbf')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict=linear_svr.predict(X_test)

print("SVR is  ",linear_svr.score(X_test,y_test))
# print("R-sq",r2_score(y_test,lr_y_predict))
print("mean squrae error linear is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print("mean absoliate error ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
y_pred2 = lr.predict([[0,0,0,0,1,0,0,1,0,0,1,0,112,63,60,142,116,156,134,142,55,42,153,150,155,127]])
print(y_pred2)

vec =DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_y_predict=rf.predict(X_test)

print("linearRegression is  ",rf.score(X_test,y_test))
print("R-sq",r2_score(y_test,rf_y_predict))
print("mean squrae error linear is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rf_y_predict)))
print("mean absoliate error ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rf_y_predict)))


