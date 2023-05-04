from collections import namedtuple
from collections import OrderedDict

from typing import List, Union, OrderedDict, Tuple 

class SparseFeature(
    namedtuple(
        'SparseFeature', 
        ['name', 'vocabulary_size', 'embedding_dim', 'dtype', 'embedding_name', 'group_name']
    )
):
    # to deny the creation of dict to save memory
    # see: https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ()
    
    def __new__(cls, name, vocabulary_size, embedding_dim, embedding_name, group_name, dtype="float32"):
        
        assert name, "Please provide the name!"
        assert vocabulary_size, "Please provide the vocabulary_size!"
        assert embedding_dim, "Please provide the embedding_dim!"
        assert embedding_name, "Please provide the embedding_name!"
        assert group_name, "Please provide the group_name!"
        
        # will need to do the assert here for dtype
        
        return super(SparseFeature, cls).__new__(cls, name, vocabulary_size, embedding_dim, dtype, embedding_name, group_name)
    
    
class DenseFeature(
    namedtuple(
        'DenseFeature', 
        ['name', 'dimension', 'dtype']
    )
):
    # to deny the creation of dict to save memory
    # see: https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ()
    
    def __new__(cls, name, dimension, dtype="float32"):
        
        assert name, "Please provide the name!"
        assert dimension, "Please provide the dimension!"
        
        # will need to do the assert here for dtype
        
        return super(DenseFeature, cls).__new__(cls, name, dimension, dtype)
    
    
def build_input_feature_column_index(
    features: List[Union[DenseFeature, SparseFeature]]
) -> OrderedDict[str, Tuple[int, int]]:
    """
    Build a feature column index
    
    Args:
        a list of features
    
    Returns:
        OrderedDict: {feature_name: [start, end]}
    """
    
    feature_col_index = OrderedDict()
    start = 0
    for feature in features:
        name = feature.name
        
        if name in feature_col_index:  # already processed
            continue
        
        if isinstance(feature, SparseFeature):
            # for sparse feature, we uses the value for look-up
            feature_col_index[name] = (start, start + 1)
            start += 1
        elif isinstance(feature, DenseFeature):
            # for dense feature, it can be a list of floats / int
            # so we need to know the dimension
            feature_col_index[name] = (start, start + feature.dimension)
            start += feature.dimension
            
    return feature_col_index
    
    