import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
'''
    There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
'''          
import numpy as np
import pandas as pd
#from lenskit.algorithms import als
import customizedALS
import setpath
import time
import pickle
from load_npz import load_trainset_npz
import lenskit.batch as lkb
from itertools import product


if __name__ == '__main__':

    print('Test batch prediction')   
    f_import = open('../model/implictMF.pkl', 'rb')
    algo_trained = pickle.load(f_import)
    f_import.close()
    
    items = algo_trained.item_index_.to_numpy()
    users = algo_trained.user_index_.to_numpy()
        # return np.ndarray
        
    #test_users = users[0:10]
    #test_items = items[0:100]   
    #UIpairs_df = pd.DataFrame(product(test_users, test_items))
    num_users = len(users)
    all_items_one_user_resampled_std_df = pd.DataFrame(items, columns = ['item']) 
    ave_std_df = pd.DataFrame(items, columns = ['item']) 
    ave_std_df['ave_std'] = 0
    
    numResampledModels = 20
    # print(denominator)
    start = time.time()
    start_position = int(input('Enter the start position in the users array: '))
    stop_position = int(input('Enter the stop position in the users array: '))
        
    if stop_position > num_users:
        stop_position = num_users
        
    for i in range(start_position, stop_position):
        user = users[i:i+1]
        UIpairs_df = pd.DataFrame(product(user, items))
        UIpairs_df.columns = ['user', 'item']
        all_items_resampled_scores = pd.DataFrame({'score_resample1': []})
            # all_items_resampled_scores.empty should return True
        # all_items_resampled_scores = []
        
        for j in range(numResampledModels):
            print('user: %d,  sample: %d' % (i+1, j+1), end = '\r')
            f_import = open('../model/resampled_implictMF' + str(j+1) + '.pkl', 'rb')
            algo_resampled = pickle.load(f_import)
            f_import.close()
            res = lkb.predict(algo_resampled, UIpairs_df)
            # take algo and user-item pairs in ['user', 'item'] dataframe
            # refer lkpy-11/lkpy-master/tests/test_batch_predict
            # return: ['user', 'item', 'prediction']
            col_score = 'score_sample' + str(j+1)
            all_items_resampled_scores[col_score] = res['prediction']
            # ['score_sample1', 'score_sample2', ..., 'score_sample20']
        all_items_one_user_resampled_std_df['std'] = np.nanstd(all_items_resampled_scores, axis = 1)
        ave_std_df['ave_std'] = (ave_std_df['ave_std'] * j + all_items_one_user_resampled_std_df['std']) / (j + 1) 
    
    
    ave_std_df['users_counted']  = stop_position - start_position
    data_path = setpath.setpath()
    ave_std_df.to_csv(data_path + 'averaged_item_std_implicitMF_start_at_' + str(start_position) + '.csv', index = False)    
    print('Batch prediction done in %.0f seconds.\n' % (time.time() - start))
    print(ave_std_df.head())
        