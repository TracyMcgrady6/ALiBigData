import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


user_view = pd.read_csv('/users/t-mac/desktop/data_group/dataset/user_view.txt')
user_view.time_stamp = user_view.time_stamp.str[:10]
shop_info = pd.read_csv('/users/t-mac/desktop/data_group/shop_info.txt')


customer_flow = user_view.groupby(['shop_id', 'time_stamp']).size()
#flow = customer_flow.loc[1, '2016-10-25':'2016-10-31']
# print(customer_flow)


# print(customer_flow[1]) id 为1的shop
print(customer_flow[1])

print(shop_info.values[22])
customer_flow[9].plot()
plt.show()

