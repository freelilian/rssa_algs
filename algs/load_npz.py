import numpy as np
import pandas as pd

def load_trainset_npz(full_path, attri_name):
    '''
        load the pre-saved npz file of the movie ratings
    '''
    model_loaded = np.load(full_path)
    data = model_loaded['dataset']
        # numpy.ndarray
    trainset = pd.DataFrame(data, columns = attri_name)
        # dataframe
    
    trainset = trainset.astype({'user': int, 'item': int, 'rating': float, 'timestamp': int})
        # ['user', 'item', 'rating', 'timestamp']
    
    return trainset