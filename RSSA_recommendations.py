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
    
    a = 0.2
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
    
    
############################################################################################
### Import the pre-trained MF model
def import_trained_model(model_path):
    model_path = '../model/'
    f_import = open(model_path + 'implictMF.pkl', 'rb')
    trained_model = pickle.load(f_import)
    f_import.close()

    return trained_model
    
    
def get_dummy_liveUser_ratings(liveUserID):  
    testing_path = '../testing_rating_rated_items_extracted/ratings_set4_rated_only_'
    fullpath_test =  testing_path + liveUserID + '.csv'
    ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
    # print(ratings_liveUser)
        #['item', 'title', 'rating']
    new_ratings = pd.Series(ratings_liveUser.rating.to_numpy(), index = ratings_liveUser.item)
        # a series
    # Extract the not-rated-yet items
    rated_items = ratings_liveUser.item.unique() 
        # np.ndarray
        
    return new_ratings, rated_items
    
      
def get_RSSA_preds(liveUserID):
    data_path = '../data/'
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')    
    model_path = '../model/'
    trained_model = import_trained_model(model_path)
    [new_ratings, rated_items] = get_dummy_liveUser_ratings(liveUserID)
        #a series; np.ndarray
        
    ### Predicting
    [RSSA_preds, liveUser_feature] = RSSA_live_prediction(trained_model, liveUserID, new_ratings, item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: ndarray
    RSSA_preds_noRatedItems = RSSA_preds[~RSSA_preds['item'].isin(rated_items)]
        # ['item', 'score', 'count', 'rank', 'discounted_score']     

    return RSSA_preds_noRatedItems 
    
    
def get_RSSA_topN(liveUserID):
    numRec = 10
    RSSA_preds_noRatedItems = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    traditional_preds_sorted = RSSA_preds_noRatedItems.sort_values(by = 'score', ascending = False)
    discounted_preds_sorted = RSSA_preds_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
    recs_topN_traditional = traditional_preds_sorted.head(numRec)
    recs_topN_discounted = discounted_preds_sorted.head(numRec)
    
    return recs_topN_discounted.item.unique()
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)


def get_RSSA_hate_items(liveUserID):
    numRec = 10
    RSSA_preds_noRatedItems = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    data_path = '../data/'
    ave_item_score = pd.read_csv(data_path + 'averaged_item_score_implicitMF.csv')
        # ['item', 'ave_score', 'ave_discounted_score']
    RSSA_preds_noRatedItems_with_ave = pd.merge(RSSA_preds_noRatedItems, ave_item_score, how = 'left', on = 'item')
        # ['item', 'score', 'count', 'rank', 'discounted_score', 'ave_score', 'ave_discounted_score']
    RSSA_preds_noRatedItems_with_ave['margin_discounted'] = RSSA_preds_noRatedItems_with_ave['ave_discounted_score'] - RSSA_preds_noRatedItems_with_ave['discounted_score']
    RSSA_preds_noRatedItems_with_ave['margin'] = RSSA_preds_noRatedItems_with_ave['ave_score'] - RSSA_preds_noRatedItems_with_ave['score']
        # ['item', 'score', 'count', 'rank', 'discounted_score', 'ave_score', 'ave_discounted_score', 'margin_discounted', 'margin']
    recs_hate_items = RSSA_preds_noRatedItems_with_ave.sort_values(by = 'margin', ascending = False).head(numRec)
    recs_hate_items_discounted = RSSA_preds_noRatedItems_with_ave.sort_values(by = 'margin_discounted', ascending = False).head(numRec)
    
    return recs_hate_items_discounted.item.unique()
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)
    
        
def get_RSSA_hip_items(liveUserID):
    numRec = 10
    numTopN = 1000
    RSSA_preds_noRatedItems = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    RSSA_preds_noRatedItems_sort_by_score = RSSA_preds_noRatedItems.sort_values(by = 'score', ascending = False)
    RSSA_preds_noRatedItems_sort_by_Dscore = RSSA_preds_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
    RSSA_preds_noRatedItems_sort_by_score_numTopN = RSSA_preds_noRatedItems_sort_by_score.head(numTopN)
    RSSA_preds_noRatedItems_sort_by_Dscore_numTopN = RSSA_preds_noRatedItems_sort_by_Dscore.head(numTopN)
    recs_hip_items = RSSA_preds_noRatedItems_sort_by_score_numTopN.sort_values(by = 'count', ascending = True).head(numRec)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    recs_hip_items_discounted = RSSA_preds_noRatedItems_sort_by_Dscore_numTopN.sort_values(by = 'count', ascending = True).head(numRec)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    
    return recs_hip_items_discounted.item.unique()
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)    


def get_RSSA_no_clue_items(liveUserID):
    numRec = 10
    model_path = '../model/'
    [new_ratings, rated_items] = get_dummy_liveUser_ratings(liveUserID)
        #a series; np.ndarray
    data_path = '../data/'
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')    
        
    resampled_preds_high_std = high_std(model_path, liveUserID, new_ratings, item_popularity)
        # ['item', 'std', 'count', 'rank'] 
    resampled_preds_high_std_noRated = resampled_preds_high_std[~resampled_preds_high_std['item'].isin(rated_items)]
    resampled_preds_high_std_noRated_sorted = resampled_preds_high_std_noRated.sort_values(by = 'std', ascending = False)
    recs_no_clue_items = resampled_preds_high_std_noRated_sorted.head(numRec)
        # ['item', 'std', 'count', 'rank'] 
    
    return recs_no_clue_items.item.unique()
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset) 

def get_RSSA_controversial_items(liveUserID):
    numRec = 10    
    data_path = '../data/'
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')    
    model_path = '../model/'
    trained_model = import_trained_model(model_path)
    umat = trained_model.user_features_
    users = trained_model.user_index_
        # trained_model has this attributes：
            # user_features_(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
            # user_index_(pandas.Index): Users in the model (length=:math:`n`).
            
    [new_ratings, rated_items] = get_dummy_liveUser_ratings(liveUserID)
        #a series; np.ndarray
        
    ### Predicting
    [_, liveUser_feature] = RSSA_live_prediction(trained_model, liveUserID, new_ratings, item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: ndarray
    distance_method = 'cosine'
    numNeighbors = 20
    neighbors = find_neighbors(umat, users, liveUser_feature, distance_method, numNeighbors)
        # ['user', 'distance']     
    variance_neighbors = controversial(trained_model, neighbors.user.unique(), item_popularity)
        # ['item', 'variance', 'count', 'rank']    
    variance_neighbors_noRated =  variance_neighbors[~variance_neighbors['item'].isin(rated_items)]
    variance_neighbors_noRated_sorted =  variance_neighbors_noRated.sort_values(by = 'variance', ascending = False)
    recs_controversial_items = variance_neighbors_noRated_sorted.head(numRec)

    return recs_controversial_items.item.unique()
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset) 

if __name__ == "__main__":
    print('Here are some dummy live users with about 20 ratings each: ')
    print('    Bart, Daricia, Sushmita, Shahan, Aru, Mitali, Yash')
    liveUserID = input('\nEnter a user ID: ')
    
    rec_ids_topn = get_RSSA_topN(liveUserID)
    print(rec_ids_topn)
    
    rec_ids_hate_items = get_RSSA_hate_items(liveUserID)
    print(rec_ids_hate_items)
    
    rec_ids_hip_items = get_RSSA_hip_items(liveUserID)
    print(rec_ids_hip_items)
    
    rec_ids_no_clue_items = get_RSSA_no_clue_items(liveUserID)
    print(rec_ids_no_clue_items)
    
    rec_ids_controversial_items = get_RSSA_controversial_items(liveUserID)
    print(rec_ids_controversial_items)
    
    