from tool import *
import pandas as pd
import numpy as np
from numpy import *
user_pay_count = pd.read_csv('/users/t-mac/desktop/data_group/dataset/shop_pay_count.csv')

a=np.array([0,0,0,1,1,1,1,1,9,2,3])

aa=pd.DataFrame(list(set(a)))
aa.to_csv('/users/t-mac/desktop/11111.csv')