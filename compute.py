"""
compute.py
"""
from typing import List
from models import Rating, Preference

import algs.RSSA_recommendations as RSSA
import pandas as pd
import numpy as np

def get_predictions(ratings: List[Rating], user_id) -> pd.DataFrame:
    # Could we also put the item_popularity.csv into database as well?
    data_path = './algs/data/'
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv')    
    
    model_path = './algs/model/'
    trained_model = RSSA.import_trained_model(model_path)
    
    new_ratings = pd.Series(rating.rating for rating in ratings)
    rated_items = np.array([np.int64(rating.item_id) for rating in ratings])
    
    ### Predicting
    [RSSA_preds, liveUser_feature] = RSSA.RSSA_live_prediction(trained_model, user_id, new_ratings, item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: ndarray    
    RSSA_preds_noRatedItems = RSSA_preds[~RSSA_preds['item'].isin(rated_items)]
        # ['item', 'score', 'count', 'rank', 'discounted_score']      
        
    return RSSA_preds_noRatedItems     


def predict_user_topN(ratings: List[Rating], user_id) -> List[Preference]:
    numRec = 10
    RSSA_preds_noRatedItems = get_predictions(ratings, user_id)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    traditional_preds_sorted = RSSA_preds_noRatedItems.sort_values(by = 'score', ascending = False)
    discounted_preds_sorted = RSSA_preds_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
    recs_topN_traditional = traditional_preds_sorted.head(numRec)
    recs_topN_discounted = discounted_preds_sorted.head(numRec)
     
    recommendations = []
    for index, row in recs_topN_discounted.iterrows():
        recommendations.append(Preference(str(np.int64(row['item'])), 'topN'))

    return recommendations

def predict_user_hate_items(ratings: List[Rating], user_id) -> List[Preference]:
    numRec = 10
    RSSA_preds_noRatedItems = get_predictions(ratings, user_id)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    data_path = './algs/data/'
    ave_item_score = pd.read_csv(data_path + 'averaged_item_score_implicitMF.csv')
        # ['item', 'ave_score', 'ave_discounted_score']
    RSSA_preds_noRatedItems_with_ave = pd.merge(RSSA_preds_noRatedItems, ave_item_score, how = 'left', on = 'item')
        # ['item', 'score', 'count', 'rank', 'discounted_score', 'ave_score', 'ave_discounted_score']
    RSSA_preds_noRatedItems_with_ave['margin_discounted'] = RSSA_preds_noRatedItems_with_ave['ave_discounted_score'] - RSSA_preds_noRatedItems_with_ave['discounted_score']
    RSSA_preds_noRatedItems_with_ave['margin'] = RSSA_preds_noRatedItems_with_ave['ave_score'] - RSSA_preds_noRatedItems_with_ave['score']
        # ['item', 'score', 'count', 'rank', 'discounted_score', 'ave_score', 'ave_discounted_score', 'margin_discounted', 'margin']
    recs_hate_items = RSSA_preds_noRatedItems_with_ave.sort_values(by = 'margin', ascending = False).head(numRec)
    recs_hate_items_discounted = RSSA_preds_noRatedItems_with_ave.sort_values(by = 'margin_discounted', ascending = False).head(numRec)
    
    recommendations = []
    for index, row in recs_hate_items_discounted.iterrows():
        recommendations.append(Preference(str(np.int64(row['item'])), 'hateItems'))

    return recommendations
    
    
def predict_user_hip_items(ratings: List[Rating], user_id) -> List[Preference]:
    numRec = 10
    RSSA_preds_noRatedItems = get_predictions(ratings, user_id)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    numTopN = 1000    
    RSSA_preds_noRatedItems_sort_by_score = RSSA_preds_noRatedItems.sort_values(by = 'score', ascending = False)
    RSSA_preds_noRatedItems_sort_by_Dscore = RSSA_preds_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
    RSSA_preds_noRatedItems_sort_by_score_numTopN = RSSA_preds_noRatedItems_sort_by_score.head(numTopN)
    RSSA_preds_noRatedItems_sort_by_Dscore_numTopN = RSSA_preds_noRatedItems_sort_by_Dscore.head(numTopN)
    recs_hip_items = RSSA_preds_noRatedItems_sort_by_score_numTopN.sort_values(by = 'count', ascending = True).head(numRec)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    recs_hip_items_discounted = RSSA_preds_noRatedItems_sort_by_Dscore_numTopN.sort_values(by = 'count', ascending = True).head(numRec)
        # ['item', 'score', 'count', 'rank', 'discounted_score']     
        
    recommendations = []
    for index, row in recs_hip_items_discounted.iterrows():
        recommendations.append(Preference(str(np.int64(row['item'])), 'hipItems'))

    return recommendations
    
    
def predict_user_no_clue_items(ratings: List[Rating], user_id) -> List[Preference]:
    new_ratings = pd.Series(rating.rating for rating in ratings)
    rated_items = np.array([np.int64(rating.item_id) for rating in ratings])

    numRec = 10
    data_path = './algs/data/'
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv') 
    model_path = './algs/model/'
    resampled_preds_high_std = RSSA.high_std(model_path, liveUserID, new_ratings, item_popularity)
        # ['item', 'std', 'count', 'rank'] 
    resampled_preds_high_std_noRated = resampled_preds_high_std[~resampled_preds_high_std['item'].isin(rated_items)]
    resampled_preds_high_std_noRated_sorted = resampled_preds_high_std_noRated.sort_values(by = 'std', ascending = False)
    recs_no_clue_items = resampled_preds_high_std_noRated_sorted.head(numRec)

    recommendations = []
    for index, row in recs_no_clue_items.iterrows():
        recommendations.append(Preference(str(np.int64(row['item'])), 'noClue'))

    return recommendations
    
    
def predict_user_controversial_items(ratings: List[Rating], user_id) -> List[Preference]:
    new_ratings = pd.Series(rating.rating for rating in ratings)
    rated_items = np.array([np.int64(rating.item_id) for rating in ratings])

    numRec = 10
    data_path = './algs/data/'
    item_popularity = pd.read_csv(data_path + 'item_popularity.csv') 
    model_path = './algs/model/'
    trained_model = RSSA.import_trained_model(model_path)
    umat = trained_model.user_features_
    users = trained_model.user_index_
        # trained_model has this attributes：
            # user_features_(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
            # user_index_(pandas.Index): Users in the model (length=:math:`n`).
    ### Predicting
    [_, liveUser_feature] = RSSA.RSSA_live_prediction(trained_model, liveUserID, new_ratings, item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: ndarray
    distance_method = 'cosine'
    numNeighbors = 20
    neighbors = RSSA.find_neighbors(umat, users, liveUser_feature, distance_method, numNeighbors)
        # ['user', 'distance']     
    variance_neighbors = RSSA.controversial(trained_model, neighbors.user.unique(), item_popularity)
        # ['item', 'variance', 'count', 'rank']    
    variance_neighbors_noRated =  variance_neighbors[~variance_neighbors['item'].isin(rated_items)]
    variance_neighbors_noRated_sorted =  variance_neighbors_noRated.sort_values(by = 'variance', ascending = False)
    recs_controversial_items = variance_neighbors_noRated_sorted.head(numRec)
    
    
    recommendations = []
    for index, row in recs_controversial_items.iterrows():
        recommendations.append(Preference(str(np.int64(row['item'])), 'controversialItems'))

    return recommendations
    
    

if __name__ == '__main__':
    testing_path = './algs/testing_rating_rated_items_extracted/ratings_set6_rated_only_'
    liveUserID = 'Bart'
    fullpath_test =  testing_path + liveUserID + '.csv'
    ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
    
    ratings = []
    for index, row in ratings_liveUser.iterrows():
        ratings.append(Rating(row['item'], row['rating']))
    
    recommendations = predict_user_topN(ratings, liveUserID)
    print(recommendations)
    print()
    
    recommendations = predict_user_hate_items(ratings, liveUserID)
    print(recommendations)
    print()
    
    recommendations = predict_user_hip_items(ratings, liveUserID)
    print(recommendations)
    print()
    
    recommendations = predict_user_no_clue_items(ratings, liveUserID)
    print(recommendations)
    print()
    
    recommendations = predict_user_controversial_items(ratings, liveUserID)
    print(recommendations)
    print()
        
    