import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

_path = "/Users/tien/Documents/PythonEnvs/pytorch/jup/recsys_models/data/short_video_understanding/byterec_sample.txt"

def read_data(path=_path):
        return pd.read_csv(
            path, 
            sep='\t',
            names=["uid", "user_city", "item_id", "author_id", 
                   "item_city", "channel", "finish", "like",
                   "music_id", "device", "time", "duration_time"])
        
def get_targets():
    return ['finish', 'like']

def process_features(data, sparse_features, dense_features):
    """
        data: Pandas dataframe
        sparse_features: sparse features names
        dense_features: dense features names
    """
    # label encoding for sparse features
    # 
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    # normalizing features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    
    return data   