import os

def set_data_path(subdirectory = './data/'):
    working_path = os.path.join(os.path.dirname(__file__), subdirectory)
    return working_path
    
def set_model_path(subdirectory = './model/'):
    working_path = os.path.join(os.path.dirname(__file__), subdirectory)
    return working_path
  