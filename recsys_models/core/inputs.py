from collections import OrderedDict
from typing import List, Union

import torch
from torch import Tensor, nn

from jup.recsys_models.core.utils import concat_fun
from jup.recsys_models.features import (DenseFeature, SparseFeature,
                                        VarLenSparseFeature,
                                        get_sparse_feature)


def create_embedding_matrix(
    features,
    init_stds=0.0001,
    # linear=False,
    # sparse=False, # used for the nn.Embedding
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

    sparse_features = get_sparse_feature(features=features)

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
        tensor: Tensor
        assert tensor
        nn.init.normal_(tensor.weight, mean=0, std=init_stds)

    return embedding_dict.to(device)


def get_feature_names(
    features: List[Union[DenseFeature, SparseFeature, VarLenSparseFeature]]
) -> List:
    features_dict = build_input_feature_column_index(features)
    return list(features_dict.keys())


def build_input_feature_column_index(
    features: List[Union[DenseFeature, SparseFeature]]
) -> OrderedDict:
    """Build a feature column index

    Args:
        a list of features

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


def combine_dnn_input(sparse_embedding_list, dense_value_list):
    """Combines all the provided list into one layer of value

    Args:
        ## to be filled

    Returns:
        ## to be filled

    Notes:
        ## will need to write unittest for this to understand this better.
    """
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError
