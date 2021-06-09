import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
'''
    There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
'''          
import numpy as np
import pandas as pd
from lenskit.algorithms import als
import time
import setpath
import pickle
from load_npz import load_trainset_npz

if __name__ == "__main__":
    ## load users that covered in the trainset
    # There are two ways to get the user IDs
        # 1. load the trainset and get the unique users, the trainset is 125443kb
        # 2. load the trained MF model, and get the attribute: ._user_index_; the trained model is 35910kb
    model_path = '../model/'
    f_import = open(model_path + 'implictMF.pkl', 'rb')
    algo = pickle.load(f_import)
    f_import.close()
    
    users = algo.user_index_.to_numpy()
    num_users = len(users)
        # 161320 users
    items = algo.item_index_.to_numpy()
    ## items: ndarray -> df
    # The 'items' here is the item IDs of the trainset of RSSA dataset
    all_items_resampled_preds_df = pd.DataFrame(items, columns = ['item']) 
    all_items_all_users_std_df = pd.DataFrame(items, columns = ['item']) 
    all_items_ave_std_df = pd.DataFrame(items, columns = ['item']) 
    
    ## load resampled models and predict for each user with each model
    count_users = 0
    start = time.time()
    for user in users[0:100]:
        count_users += 1;
        print(num_users - (count_users + 1), end = '\r') 
        
        numResampledModels = 20
        for i in range(numResampledModels):
            filename = model_path + 'resampled_implictMF' + str(i + 1) + '.pkl'
            f_import = open(filename, 'rb')
            algo = pickle.load(f_import)
            f_import.close()
            
            ## do not apply the discounting method here on the output side
            # RSSA_preds = RSSA_live_prediction(algo, liveUserID, new_ratings, item_popularity)
                # ['item', 'score', 'count', 'rank', 'discounted_score']
            
            #!!! items should be different in differend resampled models
            # algo has this attribute: 
                # item_index_(pandas.Index): Items in the model (length=:math:`n`).
            items_in_sample = algo.item_index_.to_numpy()
                # only include items resampled in this current model
            resampled_preds = algo.predict_for_user(user, items_in_sample)
                # return a series with 'items' as the index
            resampled_preds_df = resampled_preds.to_frame().reset_index()
            col = 'score' + str(i+1)
            resampled_preds_df.columns = ['item', col]
            all_items_resampled_preds_df = pd.merge(all_items_resampled_preds_df, resampled_preds_df, how = 'left', on = 'item')
            
        ## calculate std 
        # numpy.nanstd
        # Compute the standard deviation along the specified axis, while ignoring NaNs.
        preds_only_df = all_items_resampled_preds_df.drop(columns=['item'])
            # preds_only_df = all_items_resampled_preds_df.drop(['item'], axis=1)
        all_items_resampled_preds_df['std'] = np.nanstd(preds_only_df, axis = 1)
            # Compute the arithmetic std along the specified axis, ignoring NaNs.
            # axis = 1 horizontlely
            # axis = 0 vertically
            # ['item', 'score1', 'score2', ... 'score20', 'std']
        # print(all_items_resampled_preds_df['std'])
        
        all_items_all_users_std_df[str(user)] = all_items_resampled_preds_df['std']
    
    preds_only_all_users_df = all_items_all_users_std_df.drop(columns=['item'])
    all_items_ave_std_df['averaged_std'] = np.nanmean(preds_only_all_users_df, axis = 1)
        # Compute the arithmetic std along the specified axis, ignoring NaNs.
        # axis = 1 horizontlely
        # axis = 0 vertically
    print("\nIt took %.0f seconds to calculate the averaved item scores." % (time.time() - start))  
    
    data_path = setpath.setpath()
    all_items_ave_std_df.to_csv(data_path + 'averaged_resampled_std.csv', index = False)    
        
        
        
        
        
        
        