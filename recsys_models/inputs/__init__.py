from collections import OrderedDict
from typing import List, Union

from torch import nn

from jup.recsys_models.features import (DenseFeature, SparseFeature,
                                        VarLenSparseFeature)


def build_input_feature_column_index(
    features: List[Union[DenseFeature, SparseFeature]]
) -> OrderedDict:
    """Build a feature column index

    Args:
        features: list of features

    Returns:
        OrderedDict: {feature_name: [start, end]}

    # Corresponding to build_input_features
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
        else:
            raise NotImplementedError(
                "We have not supported VarLenSparseFeature and other types!"
            )

    return feature_col_index


def create_embedding_matrix(
    sparse_features,
    init_stds=0.0001,
    device="cpu",
) -> nn.ModuleDict:
    """
    It is a common practice that for sparse features, we create
    embedding dictionary for each sparse feature.

    Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}

    Note: we are not dealing with VarLenSparse Feature at this moment

    To make it simple, we wont support linear and sparse args here as well.
    """

    # confirm that there is no dense features before we can build out
    # embeddings for sparse features
    # we also confirm that we wont support VarLen at this time.
    # we only extract sparse feature
    # we do not support dense feature at this time

    for feature in sparse_features:
        assert isinstance(feature, SparseFeature), "we only create embedding for sparse features!"

    # reading:
    # https://medium.com/@gautam.e/what-is-nn-embedding-really-de038baadd24
    embedding_dict = nn.ModuleDict(
        {
            feature.embedding_name: nn.Embedding(
                feature.vocabulary_size, feature.embedding_dim
            )
            for feature in sparse_features
        }
    )

    # Need to figure out why the red line here
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_stds)

    return embedding_dict.to(device)


# this has to be in a base file 
def compute_input_dim(features, include_sparse=True, include_dense=True):
    # TODO (@tien): figure out what feature_group is for
    
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