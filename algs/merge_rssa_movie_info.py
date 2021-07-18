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
#import customizedALS
import time
import setpath
import pickle
from scipy.spatial.distance import cosine

if __name__ == "__main__":    
    ### Import item popularity for discounting from the outout side
    data_path = setpath.setpath()
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')
        # ['item', 'count', 'rank'], sorted by rank
        
    # Read movie info dataset including movie titles 
    movie_info_path = '../../data/ml-25m20m_movie_info_poster.csv'
    movie_info = pd.read_csv(movie_info_path, encoding='latin1')
    movie_info = movie_info.rename({'movie_id': 'item'}, axis = 1)
    
    rssa_movie_info = pd.merge(item_popularity, movie_info, how = 'left', on = 'item')
    rssa_movie_info = rssa_movie_info[['item', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster', 'count', 'rank']]
    rssa_movie_info = rssa_movie_info.rename({'item': 'movie_id'}, axis = 1)
    rssa_movie_info.to_csv('../data/rssa_movie_info.csv', index = False)