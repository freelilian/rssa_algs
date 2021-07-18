# RSSA recommendation lists

import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
         
import numpy as np
import pandas as pd
import time
import setpath
import pickle
from scipy.spatial.distance import cosine

def RSSA_live_prediction(algo, liveUserID, new_ratings, item_popularity):    
    '''
    algo: trained implicitMF model
    liveUserID: str
    new_ratings: Series
    N: # of recommendations
    item_popularity: ['item', 'count', 'rank']
    '''
    items = item_popularity.item.unique()
        # items is NOT sorted
    #>>> items, rating_counts = np.unique(ratings_train['item'], return_counts = True)
        # items is sorted by default
    als_implicit_preds, liveUser_feature = algo.predict_for_user(liveUserID, items, new_ratings)
        # return a series with 'items' as the index & liveUser_feature: np.ndarray
    als_implicit_preds_df = als_implicit_preds.to_frame().reset_index()
    als_implicit_preds_df.columns = ['item', 'score']
    # print(als_implicit_preds_df.sort_values(by = 'score', ascending = False).head(10))
    
    ## discounting popular items
    highest_count = item_popularity['count'].max()
    digit = 1
    while highest_count/(10 ** digit) > 1:
        digit = digit + 1
    denominator = 10 ** digit
    # print(denominator)
    
    a = 0.5 # try [0.2, 0.3, 0.4, 0.5, 0.6, 1] on set 6
    als_implicit_preds_popularity_df = pd.merge(als_implicit_preds_df, item_popularity, how = 'left', on = 'item')
    RSSA_preds_df = als_implicit_preds_popularity_df
    RSSA_preds_df['discounted_score'] = RSSA_preds_df['score'] - a*(RSSA_preds_df['count']/denominator)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    
    # RSSA_preds_df_sorted = RSSA_preds_df.sort_values(by = 'discounted_score', ascending = False)
        
    return RSSA_preds_df, liveUser_feature    


def high_std(model_path, liveUserID, new_ratings, item_popularity):
#def high_std(model_path, liveUserID, new_ratings, item_popularity):
    
    items = item_popularity.item.unique()
    ## items: ndarray -> df
    # The 'items' here is the item IDs of the trainset of RSSA dataset
    all_items_resampled_preds_df = pd.DataFrame(items, columns = ['item'])
    
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
            # np.ndarray
        resampled_preds, _ = algo.predict_for_user(liveUserID, items_in_sample, new_ratings)
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
    all_items_std_df = all_items_resampled_preds_df[['item', 'std']]
        # ['item', 'std']
    all_items_std_df = pd.merge(all_items_std_df, item_popularity, how = 'left', on = 'item')
        # ['item', 'std', 'count', 'rank'] 
    
    return all_items_std_df
    
    
def controversial(algo, users_neighbor, item_popularity):
    items = item_popularity.item.unique()
        # items is NOT sorted
    neighbor_scores_df = pd.DataFrame(items, columns = ['item'])
    for neighbor in users_neighbor:
        neighbor_implicit_preds = algo.predict_for_user(neighbor, items)
            # return a series with 'items' as the index
        neighbor_implicit_preds_df = neighbor_implicit_preds.to_frame().reset_index()
        neighbor_implicit_preds_df.columns = ['item', 'score']
        neighbor_scores_df[str(neighbor)] = neighbor_implicit_preds_df['score']
    
    neighbor_scores_only_df = neighbor_scores_df.drop(columns = ['item'])
    neighbor_scores_df['variance'] = np.nanvar(neighbor_scores_only_df, axis = 1)
        # ['item', 'neighbor1', 'neighbor2', ..., 'neighborN', 'variance']    
    neighbor_variance_df = neighbor_scores_df[['item', 'variance']]
        # ['item', 'variance']
    neighbor_variance_df = pd.merge(neighbor_scores_df, item_popularity, how = 'left', on = 'item')
        # ['item', 'variance', 'count', 'rank']  
        
    return neighbor_variance_df
         
        
def similarity_user_features(umat, users, feature_newUser, method = 'cosine'):
    '''
        ALS has already pre-weighted the user features/item features;
        Use either the Cosine distance(by default) or the Eculidean distance;
        umat: np.ndarray
        users: Int64Index
        feature_newUser: np.ndarray
    '''        
    nrows, ncols = umat.shape
    # distance = np.zeros([1, nrows])
    distance = []
    if method == 'cosine':
        for i in range(nrows):
            feature_oneUser = umat[i,]
            dis = cosine(feature_oneUser, feature_newUser)
            distance.append(dis)
    elif method == 'eculidean':
        for i in range(nrows):
            feature_oneUser = umat[i,]
            dis = np.linalg.norm(feature_oneUser-feature_newUser)
                # This works because Euclidean distance is l2 norm and 
                # the default value of ord parameter in numpy.linalg.norm is 2.
            distance.append(dis)
    # convert to a dataframe with indexing of items
    # print(users)
        # Int64Index
    distance = pd.DataFrame({'user': users.values, 'distance': distance})
    #print(distance.head())

    return distance

def find_neighbors(umat, users, feature_newUser, distance_method, num_neighbors):
    similarity = similarity_user_features(umat, users, feature_newUser, distance_method)
        # ['user', 'distance']
    # min_distanc = ??
    similarity_sorted = similarity.sort_values(by = 'distance', ascending = True)
    neighbors_similarity = similarity_sorted.head(num_neighbors)
        # ['user', 'distance']
    
    return neighbors_similarity
    

        
    
if __name__ == "__main__":    
    ### Import implicit MF model, saved in an object
    model_path = './model/'
    f_import = open(model_path + 'implictMF.pkl', 'rb')
    trained_model = pickle.load(f_import)
    f_import.close()

    ### Import item popularity for discounting from the output side
    data_path = setpath.setpath()
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')
        # ['item', 'count', 'rank'], sorted by rank
        
    # Read movie info dataset including movie titles 
    movie_info_path = data_path + 'rssa_movie_info.csv'
    movie_info = pd.read_csv(movie_info_path, encoding='latin1')
    movie_title = movie_info[['movie_id', 'title']] 
    movie_title = movie_title.rename({'movie_id': 'item'}, axis = 1)
        # ['item', 'title']
    # print(type(movie_title['item'][0]))
    # print(movie_title.head(10))
    

    ### Import new ratings of the live user
    # RSSA_team = ['Bart', 'Daricia', 'Sushmita', 'Shahan', 'Aru', 'Mitali', 'Yash']
    RSSA_team = ['Bart', 'Sushmita', 'Shahan', 'Aru', 'Mitali', 'Yash']
    
    for liveUserID in RSSA_team:
        # liveUserID = input('Enter a user ID: ')
        # liveUserID = 'Bart'
        testing_path = './testing_rating_rated_items_extracted/ratings_set6_rated_only_'
        fullpath_test =  testing_path + liveUserID + '.csv'
        ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
        # print(ratings_liveUser)
            #['item', 'title', 'rating']
        new_ratings = pd.Series(ratings_liveUser.rating.to_numpy(), index = ratings_liveUser.item)

        N = 10 
                
        ### Predicting
        [RSSA_preds, liveUser_feature] = RSSA_live_prediction(trained_model, liveUserID, new_ratings, item_popularity)
            # ['item', 'score', 'count', 'rank', 'discounted_score']
            # liveUser_feature: ndarray
        RSSA_preds_titled = pd.merge(RSSA_preds, movie_title, how = 'left', on = 'item')
            # ['item', 'score', 'count', 'rank', 'discounted_score', 'title']  

        # extract the not-rated-yet items
        rated_items = ratings_liveUser.item.unique()
        RSSA_preds_titled_noRated1 = RSSA_preds_titled[~RSSA_preds_titled['item'].isin(rated_items)]
             # ['item', 'score', 'count', 'rank', 'discounted_score', 'title']  
        #print(RSSA_preds_titled.shape)    
        #print(RSSA_preds_titled_noRated1.shape)  

        #### Generate recommendations
        print('\n\nRSSA recommendations for user: %s' % liveUserID)
        #===> 1 - RSSA TopN
        print('\n1 - RSSA Discounted Top-N:')
        traditional_preds_sorted = RSSA_preds_titled_noRated1.sort_values(by = 'score', ascending = False)
        discounted_preds_sorted = RSSA_preds_titled_noRated1.sort_values(by = 'discounted_score', ascending = False)
        recs_topN_traditional = traditional_preds_sorted.head(N)
        recs_topN_discounted = discounted_preds_sorted.head(N)
        #print('\nTraditional Top-N:')
        #print(recs_topN_traditional)
        # print(recs_topN_discounted[['item', 'count', 'rank', 'discounted_score', 'title']])
        print(recs_topN_discounted[['count', 'rank', 'title']])
        
        #===> 2 - Things we think you will hate
        # essential components: averaged score of each item over all the users
        print('\n2 - Things we think you will hate:')
        ave_item_score = pd.read_csv(data_path + 'averaged_item_score_implicitMF.csv')
            # ['item', 'ave_score', 'ave_discounted_score']
        RSSA_preds_titled_noRated2 = pd.merge(RSSA_preds_titled_noRated1, ave_item_score, how = 'left', on = 'item')
            # ['item', 'score', 'count', 'rank', 'discounted_score', 'title', 'ave_score', 'ave_discounted_score']
        RSSA_preds_titled_noRated2['margin_discounted'] = RSSA_preds_titled_noRated2['ave_discounted_score'] - RSSA_preds_titled_noRated2['discounted_score']
        RSSA_preds_titled_noRated2['margin'] = RSSA_preds_titled_noRated2['ave_score'] - RSSA_preds_titled_noRated2['score']
            # ['item', 'score', 'count', 'rank', 'discounted_score', 'title', 'ave_score', 'ave_discounted_score', 'margin_discounted', 'margin']
        recs_hate_items = RSSA_preds_titled_noRated2.sort_values(by = 'margin', ascending = False).head(N)
        recs_hate_items_discounted = RSSA_preds_titled_noRated2.sort_values(by = 'margin_discounted', ascending = False).head(N)
        #print(recs_hate_items[['item', 'count', 'rank', 'margin_discounted', 'title']])
        # print(recs_hate_items_discounted[['item', 'count', 'rank', 'margin_discounted', 'title']])
        print(recs_hate_items_discounted[['count', 'rank', 'title']])
        
        #===> 3 - Things you will be among the first to try
        # essential components: rating counts of each item
        numTopN = 1000
        print('\n3 - Things you will be among the first to try:')
        RSSA_preds_titled_noRated2_sort_by_score = RSSA_preds_titled_noRated2.sort_values(by = 'score', ascending = False)
        RSSA_preds_titled_noRated2_sort_by_Dscore = RSSA_preds_titled_noRated2.sort_values(by = 'discounted_score', ascending = False)
        RSSA_preds_titled_noRated2_sort_by_score_top200 = RSSA_preds_titled_noRated2_sort_by_score.head(numTopN)
        RSSA_preds_titled_noRated2_sort_by_Dscore_top200 = RSSA_preds_titled_noRated2_sort_by_Dscore.head(numTopN)
        recs_hip_items = RSSA_preds_titled_noRated2_sort_by_score_top200.sort_values(by = 'count', ascending = True).head(N)
        recs_hip_items_discounted = RSSA_preds_titled_noRated2_sort_by_Dscore_top200.sort_values(by = 'count', ascending = True).head(N)
        #print(recs_hip_items[['item', 'rank', 'discounted_score', 'count', 'title']])
        # print(recs_hip_items_discounted[['item', 'rank', 'discounted_score', 'count', 'title']])
        print(recs_hip_items_discounted[['rank', 'count', 'title']])
        
        #===> 4 - Things we have no clue about
        # essential components: resample, train 20 resampled implicitMF models offline
        # personalized std: std/ave(std): std of an item/ ave (stds) of this item over all the other users.
        print('\n4 - Things we have no clue about:')
        #print('\n  4.1 - Non-personalized list:')
        
        resampled_preds_high_std = high_std(model_path, liveUserID, new_ratings, item_popularity)
           # ['item', 'std', 'count', 'rank'] 
        resampled_preds_high_std_titled = pd.merge(resampled_preds_high_std, movie_title, how = 'left', on = 'item')
            # ['item', 'std', 'count', 'rank', 'title']
        resampled_preds_high_std_titled_norated = resampled_preds_high_std_titled[~resampled_preds_high_std_titled['item'].isin(rated_items)]
        resampled_preds_high_std_titled_norated_sorted = resampled_preds_high_std_titled_norated.sort_values(by = 'std', ascending = False)
        recs_no_clue_items = resampled_preds_high_std_titled_norated_sorted.head(N)
        # print(recs_no_clue_items[['item', 'rank', 'count', 'title']])
        print(recs_no_clue_items[['rank', 'count', 'title']])
        
        #print('\n  4.2 - Personalized list:')
        #print('\tProblematic: it is both memory expensive and computation expensive.') 
        #print('\tTwo levels iteration (161320 * 20 * 57433) & giant matrix (161320 * 57433)')
        
        
        #===> 5 - Things that are controversial
        # essential components: offline user_latent matrix; new user feature array; 
        # idea:
        # find neighbours (n = 20) based on user_latend_matrix similarity with distance
        print('\n5 - Things that are controversial:')
        num_neighbors = 20
        # algo has this attributes：
            # user_features_(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
            # user_index_(pandas.Index): Users in the model (length=:math:`n`).
        umat = trained_model.user_features_
        # np.ndarray
        # print(liveUser_feature)
        # np.ndarray 
        users = trained_model.user_index_
        # print(users)
        distance_method = 'cosine'
        neighbors = find_neighbors(umat, users, liveUser_feature, distance_method, num_neighbors)
            # ['user', 'distance'] 
        # print(neighbors.user.unique())
        variance_neighbors = controversial(trained_model, neighbors.user.unique(), item_popularity)
            # ['item', 'variance', 'count', 'rank']
        variance_neighbors_titled =  pd.merge(variance_neighbors, movie_title, how = 'left', on = 'item')       
            # ['item', 'variance', 'count', 'rank', 'title']
        variance_neighbors_titled_noRated =  variance_neighbors_titled[~variance_neighbors_titled['item'].isin(rated_items)]
        variance_neighbors_titled_noRated_sorted =  variance_neighbors_titled_noRated.sort_values(by = 'variance', ascending = False)
        recs_controversial_items = variance_neighbors_titled_noRated_sorted.head(N)
        # print(recs_controversial_items[['item', 'rank', 'count', 'variance', 'title']])
        print(recs_controversial_items[['rank', 'count', 'title']])
        print('-------------------------------------------------------------------------------------')
        '''
        num_TopN = [200, 500, 1000, 2000]
        for num in num_TopN:
            print('\n3 - Things you will be among the first to try (with topN = %d):' % num)
            RSSA_preds_titled_noRated2_sort_by_score = RSSA_preds_titled_noRated2.sort_values(by = 'score', ascending = False)
            RSSA_preds_titled_noRated2_sort_by_Dscore = RSSA_preds_titled_noRated2.sort_values(by = 'discounted_score', ascending = False)
            RSSA_preds_titled_noRated2_sort_by_score_top200 = RSSA_preds_titled_noRated2_sort_by_score.head(num)
            RSSA_preds_titled_noRated2_sort_by_Dscore_top200 = RSSA_preds_titled_noRated2_sort_by_Dscore.head(num)
            recs_hip_items = RSSA_preds_titled_noRated2_sort_by_score_top200.sort_values(by = 'count', ascending = True).head(N)
            recs_hip_items_discounted = RSSA_preds_titled_noRated2_sort_by_Dscore_top200.sort_values(by = 'count', ascending = True).head(N)
            #print(recs_hip_items[['item', 'rank', 'discounted_score', 'count', 'title']])
            print(recs_hip_items_discounted[['item', 'rank', 'discounted_score', 'count', 'title']])
        '''