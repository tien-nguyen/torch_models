from collections import namedtuple
from typing import List, Union


class SparseFeature(
    namedtuple(
        'SparseFeature',
        ['name', 'vocabulary_size', 'embedding_dim', 'embedding_name',
         'use_hash', 'dtype'])):
    
    __slots__ = ()
    
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None):
        
        if embedding_name is None:
            embedding_name = name
        
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        
        return super(SparseFeature, cls).__new__(cls, name, vocabulary_size=vocabulary_size, 
                                                 embedding_dim=embedding_dim, use_hash=use_hash, dtype=dtype,
                                                 embedding_name=embedding_name)

class DenseFeature(namedtuple('DenseFeature', ['name', 'dimension', 'dtype'])):
    
    __slots__ = ()
    
    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeature, cls).__new__(cls, name, dimension, dtype)
    
    def __hash__(self):
        return self.name.__hash__()
    

class VarLenSparseFeature(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeature, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()
    
#################################
#  FUNCTIONS                    #
#################################

def compute_input_dim(features, include_sparse=True, include_dense=True):
    
    sparse_features = list(
        filter(
            lambda x : isinstance(x, (SparseFeature, VarLenSparseFeature)),
            features
        )
    ) if len(features) else []
    
    dense_features = list(
        filter(
            lambda x : isinstance(x, DenseFeature), 
            features
        )
    ) if len(features) else []
    
    dense_feature_dim = sum(
        map(lambda x : x.dimension, dense_features)
    )
    
    sparse_input_dim = sum(feat.embedding_dim for feat in sparse_features)
    
    input_dim = 0
    
    if include_dense:
        input_dim += dense_feature_dim
    
    if include_sparse:
        input_dim += sparse_input_dim
    
    return input_dim

def get_sparse_feature(features: List[Union[SparseFeature, DenseFeature]]) -> List:
    return list(
        filter(
            lambda x : isinstance(x, SparseFeature), 
            features
        ) if len(features) else []
    )
    
    
def get_dense_feature(features: List[Union[SparseFeature, DenseFeature]]):
    return list(
        filter(
            lambda x : isinstance(x, DenseFeature),
            features
        ) if len(features) else []
    ) 