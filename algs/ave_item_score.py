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
    
def averaged_item_score(algo, transet, item_popularity, a = 0.2):  
    '''
        algo: trained implicitMF model
        transet: ['user', 'item', 'rating', 'timestamp']
        new_ratings: Series
        N: # of recommendations
        item_popularity: ['item', 'count', 'rank']
    '''
    ###
    items = transet.item.unique()
        # items is NOT sorted by derault
    users = transet.user.unique()
    num_users = len(users)
        # users is NOT sorted by derault
    # print(num_users)
        # 161320 users
    # print(users)
    
    ## discounting popular items
    highest_count = item_popularity['count'].max()
    digit = 1
    while highest_count/(10 ** digit) > 1:
        digit = digit + 1
    denominator = 10 ** digit
    # print(denominator)
    
    ## items: ndarray -> df
    ave_scores_df = pd.DataFrame(items, columns = ['item'])
    ave_scores_df['ave_score'] = 0
    ave_scores_df['ave_discounted_score'] = 0
    #print(ave_scores_df.head(20))
    #print(ave_scores_df.tail(20))
    calculated_users = -1
    start = time.time()
    for user in users:
    #for user in users[0:1000]:
        calculated_users += 1;
        print(num_users - (calculated_users + 1), end = '\r') 
            # flushing does not work
        user_implicit_preds = algo.predict_for_user(user, items)
            # the ratings of the user is already in the trainset used to train the algo
            # return a series with 'items' as the index, order is the same
        user_implicit_preds_df = user_implicit_preds.to_frame().reset_index()
        user_implicit_preds_df.columns = ['item', 'score']
        user_implicit_preds_df = pd.merge(user_implicit_preds_df, item_popularity, how = 'left', on = 'item')
            # ['item', 'score', 'count', 'rank']
        user_implicit_preds_df['discounted_score'] = user_implicit_preds_df['score'] - a*(user_implicit_preds_df['count']/denominator)
            # ['item', 'score', 'count', 'rank', 'discounted_score']
                
        ave_scores_df['ave_score'] = (ave_scores_df['ave_score'] * calculated_users + user_implicit_preds_df['score'])/(calculated_users + 1)
        ave_scores_df['ave_discounted_score'] = (ave_scores_df['ave_discounted_score'] * calculated_users + user_implicit_preds_df['discounted_score'])/(calculated_users + 1)
    #print(user_implicit_preds_df.head(20))
    #print(user_implicit_preds_df.tail(20))
    
    print(ave_scores_df.head(20))
    #print(ave_scores_df.tail(20))
    print("\nIt took %.0f seconds to calculate the averaved item scores." % (time.time() - start))
    
    return ave_scores_df
    
if __name__ == "__main__":    
    

    ### Import implicit MF model, saved in an object
    f_import = open('../model/implictMF.pkl', 'rb')
    algo = pickle.load(f_import)
    f_import.close()
    
    ### Import offline dataset, this was  also used as the transet in RSSA
    data_path = setpath.setpath()
    fullpath_trian = data_path + 'train.npz'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    ratings_train = load_trainset_npz(fullpath_trian, attri_name)
    
    ### Import item popularity for discounting from the outout side
    #data_path = setpath.setpath()
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')
    
    ave_item_score = averaged_item_score(algo, ratings_train, item_popularity)
        # ['item', 'ave_score', 'ave_discounted_score']
        # It took 2499 seconds to calculate the averaved item scores.
    ave_item_score.to_csv(data_path + 'averaged_item_score_implicitMF.csv', index = False)