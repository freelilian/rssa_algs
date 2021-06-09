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

def train_resample_models():
    pass
    
if __name__ == "__main__":
    
    data_path = setpath.setpath()
    fullpath_trian = data_path + 'train.npz'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    ratings_train = load_trainset_npz(fullpath_trian, attri_name)
    # print(ratings_train.head(10))
    
    ## 1 - Discounting the input ratings by ranking
    # 1.1 - Calculating item rating counts and popularity rank,
    # This will be used to discount the popular items from the input side
    items, rating_counts = np.unique(ratings_train['item'], return_counts = True)
        # items is sorted by default
    #>>> items = item_popularity.item.unique()
        # items is NOT sorted
    items_rating_count = pd.DataFrame({'item': items, 'count': rating_counts}, columns = ['item', 'count'])
    items_rating_count_sorted = items_rating_count.sort_values(by = 'count', ascending = False)
    items_popularity = items_rating_count_sorted
    items_popularity['rank'] = range(1, len(items_popularity)+1)
        # ['item', 'count', 'rank']

    previsous_count = 0
    previsous_rank = 0
    for index, row in items_popularity.iterrows():
        current_count = row['count']
        
        if row['count'] == previsous_count:
            row['rank'] = previsous_rank

        previsous_count = current_count
        previsous_rank = row['rank']
                
    #print(items_popularity.head(20))
    #print(items_popularity.tail(20))
        # lowest rank = 476631
          
    # 1.2 - Start to discounting the input ratings by ranking
    b = 0.4
    ratings_train_popularity = pd.merge(ratings_train, items_popularity, how = 'left', on = 'item')
    ratings_train_popularity['discounted_rating'] = ratings_train_popularity['rating']*(1-b/(2*ratings_train_popularity['rank']))
    ratings_train = ratings_train_popularity[['user', 'item', 'discounted_rating', 'timestamp']]
    # print(ratings_train.head(10))
    ratings_train = ratings_train.rename({'discounted_rating': 'rating'}, axis = 1)
    
    ## 2 - Train the *@resampled@* implicit MF model
    numObservations = ratings_train.shape[0]
    numRepetition = 20
    alpha = 0.5
    start = time.time()
    resampled_algo = customizedALS.ImplicitMF(20, iterations=10, method="lu")
    for i in range(numRepetition):
        print ('Resampling 20 MF models: %d ' % (i + 1), end = '\r')
        sampled_ratings = ratings_train.sample(n = int(numObservations * alpha), replace = False)
        resampled_algo.fit(sampled_ratings)
        filename = '../model/resampled_implictMF' + str(i + 1) + '.pkl'
        f = open(filename, 'wb')
        pickle.dump(resampled_algo, f)
        f.close() 
        
    print("\n\n%d resampled MF models trained." % numRepetition)
    end = time.time() - start
    print('\nDone!!! Time spent: %0.0fs' % end)
    # Done!!! Time spent: 575s
    

