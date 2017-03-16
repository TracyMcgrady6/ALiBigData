import pandas as pd
from pandas import Series,DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import metrics
shop_info = pd.read_csv('/users/t-mac/desktop/data_group/shop_info.txt')


user_view = pd.read_csv('/users/t-mac/desktop/data_group/dataset/user_view.txt')
user_view.time_stamp = user_view.time_stamp.str[:10]


customer_flow = user_view.groupby(['shop_id', 'time_stamp']).size()

feature = DataFrame(shop_info,columns=['per_pay','score','comment_cnt','shop_level','count_view'])
for i in range(0,2000):
    feature.loc[i,'count_view'] = int(customer_flow[i + 1].mean())
#处理缺失值
feature['per_pay'].fillna(int(feature['per_pay'].mean()),inplace=True)
feature['score'].fillna(int(feature['score'].mean()),inplace=True)
feature['comment_cnt'].fillna(int(feature['comment_cnt'].mean()),inplace=True)
feature['shop_level'].fillna(int(feature['shop_level'].mean()),inplace=True)

print('-----------')
# print(feature)

X = feature[['per_pay','score','comment_cnt','shop_level']]
Y = feature['count_view']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
linreg = LinearRegression()
print(type(y_test))
linreg.fit(X_train, y_train)
print(linreg.intercept_)#截距
print(linreg.coef_)#系数
y_pred = linreg.predict(X_test)

for i in range(1,500):
    print(X_test.index[i]+1,X_test.values[i],'-------',y_test.values[i],"，实际浏览量：",customer_flow.values[X_test.index[i]+1].mean())
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print(X_test)
