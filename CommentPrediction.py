import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import metrics

supermarket = pd.read_csv('/users/t-mac/desktop/supermarket.txt')
print(supermarket)
user_pay = pd.read_csv('/users/t-mac/desktop/data_group/dataset/user_pay.txt')
user_pay.time_stamp = user_pay.time_stamp.str[:10]
customer_flow = user_pay.groupby(['shop_id', 'time_stamp']).size()

feature = pd.DataFrame(supermarket,columns=['per_pay','score','comment_cnt','shop_level','count_pay'])

for i in supermarket['shop_id']:
    feature.loc[i,'count_pay'] = int(customer_flow[i + 1].mean())


feature['per_pay'].fillna(int(feature['per_pay'].mean()),inplace=True)
feature['score'].fillna(int(feature['score'].mean()),inplace=True)
feature['comment_cnt'].fillna(int(feature['comment_cnt'].mean()),inplace=True)
feature['shop_level'].fillna(int(feature['shop_level'].mean()),inplace=True)
feature['count_pay'].fillna(int(feature['count_pay'].mean()),inplace=True)

print(feature)

print('-----------')
X = feature[['per_pay','score','comment_cnt','shop_level']]
Y = feature['count_pay']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
linreg = LinearRegression()
print(type(y_test))
linreg.fit(X_train, y_train)
print(linreg.intercept_)#截距
print(linreg.coef_)#系数
y_pred = linreg.predict(X_test)

# for i in X_test:
#     i = 0
#     print(X_test.index[i]+1,X_test.values[i],'-------',y_test.values[i],"，实际浏览量：",customer_flow.values[X_test.index[i]+1].mean())
#     i=i+1
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

