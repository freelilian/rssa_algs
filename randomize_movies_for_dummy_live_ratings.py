# shuffle movie titles
import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
'''
    There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
'''   

import pandas as pd 
import numpy as np 
from load_npz import load_trainset_npz
import csv
import setpath



#===> load saved trainset
if __name__ == '__main__':
    data_path = setpath.setpath()
    
    # load the 【sparsity reduced】 training dataset
    fullpath_train = data_path + 'train.npz'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    trainset = load_trainset_npz(fullpath_train, attri_name)
    
    # load movie info. dataset including movie titles
    movie_info_path = data_path + '/rssa_movie_info.csv'
    movie_info = pd.read_csv(movie_info_path, encoding='latin1')    
    movie_title = movie_info[['movie_id', 'title']] 
    movie_title = movie_title.rename({'movie_id': 'item'}, axis = 1)
    
    # extract unique items （item IDs） from trainset
    #items, rating_counts = np.unique(trainset['item'], return_counts = True)
    #items_rating_count = pd.DataFrame({'item': items, 'count': rating_counts}, columns = ['item', 'count'])
    #print(items_rating_count.shape)
    
    '''
    items = trainset.item.unique()
    print(type(items))
    print('--------------------------')
    items_df = pd.DataFrame(items, columns = ['item'])
    print(items_df.head())
    print('--------------------------')
    items_series = pd.Series(items)
    print(items_series.head())
    print('--------------------------')
    items_series2df = items_series.to_frame().reset_index()
    print(items_series2df.head())
    '''
    items = trainset.item.unique()
    items_df = pd.DataFrame(items, columns = ['item'])
    
    # merge trainset with movie IDs & titles
    training_items = pd.merge(items_df, movie_title, how = 'left', on = 'item')
    offline_items = training_items[['item', 'title']]
    
    # shuffle the offline items
    # frac = 1 means: A random 100% sample of the DataFrame without replacement
    offline_items_s1 = offline_items.sample(frac = 1)
    offline_items_s2 = offline_items.sample(frac = 1)
    offline_items_s3 = offline_items.sample(frac = 1)
    offline_items_s4 = offline_items.sample(frac = 1)
    offline_items_s5 = offline_items.sample(frac = 1)
    offline_items_s6 = offline_items.sample(frac = 1)
    offline_items_s7 = offline_items.sample(frac = 1)
    offline_items_s8 = offline_items.sample(frac = 1)
    offline_items_s9 = offline_items.sample(frac = 1)
    
    print(len(offline_items_s1.item.unique()))
    print(len(offline_items_s2.item.unique()))
    print(len(offline_items_s3.item.unique()))
    print(len(offline_items_s4.item.unique()))
    print(len(offline_items_s5.item.unique()))
    print(len(offline_items_s6.item.unique()))
    print(len(offline_items_s7.item.unique()))
    print(len(offline_items_s8.item.unique()))
    print(len(offline_items_s9.item.unique()))
   

    '''
    fullpath_offline_items1 = data_path + 'offline_items_Bart.csv' 
    fullpath_offline_items2 = data_path + 'offline_items_Daricia.csv' 
    fullpath_offline_items3 = data_path + 'offline_items_Sushmita.csv' 
    fullpath_offline_items4 = data_path + 'offline_items_Shahan.csv' 
    fullpath_offline_items5 = data_path + 'offline_items_Vihang.csv' 
    fullpath_offline_items6 = data_path + 'offline_items_6.csv' 
    fullpath_offline_items7 = data_path + 'offline_items_7.csv' 
    fullpath_offline_items8 = data_path + 'offline_items_8.csv' 
    fullpath_offline_items9 = data_path + 'offline_items_9.csv' 
    offline_items_s1.to_csv(fullpath_offline_items1, index = False)
    offline_items_s2.to_csv(fullpath_offline_items2, index = False)
    offline_items_s3.to_csv(fullpath_offline_items3, index = False)
    offline_items_s4.to_csv(fullpath_offline_items4, index = False)
    offline_items_s5.to_csv(fullpath_offline_items5, index = False)
    offline_items_s6.to_csv(fullpath_offline_items6, index = False)
    offline_items_s7.to_csv(fullpath_offline_items7, index = False)
    offline_items_s8.to_csv(fullpath_offline_items8, index = False)
    offline_items_s9.to_csv(fullpath_offline_items9, index = False)
    '''

    