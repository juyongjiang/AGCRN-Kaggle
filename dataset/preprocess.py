import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train = train.drop(columns=['IsHoliday'])
save_path = './processed'
if not os.path.exists(save_path): os.makedirs(save_path)
store_num = len(train['Store'].unique())

num_dept = []
for i in tqdm(range(store_num)):
    test_store_data = test[test['Store']==(i+1)]
    test_dept_id = list(test_store_data['Dept'].unique())
    
    store_data_tmp = train[train['Store']==(i+1)]
    store_data = store_data_tmp[store_data_tmp.Dept.isin(test_dept_id)]
    train_dept_id = list(store_data['Dept'].unique())
    if not test_dept_id==train_dept_id:
        print("===>", i+1)
        print(test_dept_id)
        print(train_dept_id)
        input('check')
    dept_data = store_data[['Dept', 'Weekly_Sales']].groupby('Dept')
    # print("The number of Depts: ", len(dept_data))
    # print(len(dept_data))
    num_dept.append(len(dept_data))
    len_list = []
    for idx, (dept_id, sale_data) in enumerate(dept_data):
        len_list.append(len(sale_data))
    # print(len_list)
    # print(min(len_list), max(len_list), np.mean(len_list))
    # print(len(dept_data))
    data_np = np.zeros((len(dept_data), max(len_list))) # save data [N, T]
    for idx, (dept_id, sale_data) in enumerate(dept_data):
        start = len(sale_data['Weekly_Sales'])
        data_np[idx, -start:] = sale_data['Weekly_Sales'].values
        data_np[idx, :-start] = sale_data['Weekly_Sales'].values[0] # we use the earliest to pad previous missing value 
        # if len(data_np[idx, :-start]) != 0 :
        #     print(data_np[idx, :-start])
        #     print(sale_data['Weekly_Sales'].values)
        #     print(data_np[idx, :])
    np.save(os.path.join(save_path, f'store_{i+1}.npy'), data_np)
print("Done!")
print('The number of department in each store: ', num_dept)