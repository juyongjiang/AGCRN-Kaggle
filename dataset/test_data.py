import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm


test = pd.read_csv('./test.csv')

len_list = []
for i in tqdm(range(45)):
    test_store_data = test[test['Store']==(i+1)]
    test_dept_id = list(test_store_data['Dept'].unique())
    dept_data = test_store_data.groupby('Dept')
    print(len(test_dept_id), len(dept_data))
    len_list.append(len(test_dept_id))
print("Done!")
print('The number of department in each store: ', len_list)


a = [74, 75, 67, 75, 69, 74, 72, 73, 68, 74, 73, 71, 75, 74, 73, 73, 73, 74, 74, 76, 70, 70, 73, 74, 72, 72, 74, 73, 71, 60, 75, 73, 56, 74, 70, 57, 61, 62, 72, 74, 73, 59, 57, 59, 70]
b = [74, 75, 67, 75, 70, 74, 72, 73, 69, 75, 73, 71, 75, 74, 73, 73, 73, 75, 74, 76, 70, 70, 73, 75, 73, 72, 74, 73, 71, 60, 75, 73, 56, 75, 70, 58, 62, 62, 72, 74, 73, 60, 57, 59, 71]

for idx, (a_, b_) in enumerate(zip(a, b)):
    if a_ != b_: 
        print(idx, a_, b_)