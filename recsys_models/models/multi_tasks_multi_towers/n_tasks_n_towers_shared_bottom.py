
import os
import time
from typing import List, Mapping, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.metrics import log_loss
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from jup.recsys_models.core.inputs import (build_input_feature_column_index,
                                           combine_dnn_input,
                                           create_embedding_matrix)
from jup.recsys_models.core.mlp import DNN
from jup.recsys_models.core.prediction_layer import PredictionLayer
from jup.recsys_models.core.utils import slice_arrays
from jup.recsys_models.features import (DenseFeature, SparseFeature,
                                        compute_input_dim, get_dense_feature,
                                        get_sparse_feature)
from jup.recsys_models.models.base import BaseModel


class NTasksNTowersSharedBottom(BaseModel):
    """https://arxiv.org/pdf/1706.05098.pdf

    Creates a network of two towers where its shared the same bottom DNN

    Args
        features: An iterable containing all the features used by deep part of the model.
        bottom_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of shared bottom DNN.
        tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
        l2_reg_linear: float, L2 regularizer strength applied to linear part
        l2_reg_embedding: float, L2 regularizer strength applied to embedding vector
        l2_reg_dnn: float, L2 regularizer strength applied to DNN
        init_std: float, to use as the initialize std of embedding vector
        seed: integer, to use as random seed.
        dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        dnn_activation: Activation function to use in DNN
        dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN
        task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
        task_names: list of str, indicating the predict target of each tasks
        device: str, ``"cpu"`` or ``"cuda:0"``
        gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    """

    def __init__(
        self,
        features,
        bottom_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64,),
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_use_bn=False,
        task_types=("binary", "binary"),
        task_names=("ctr", "ctcvr"),
        device="cpu",
        gpus=None,
        tensorboard_path=None,
    ):
        # some checks
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(features) == 0:
            raise ValueError("features is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ["binary", "regression"]:
                raise ValueError(
                    "task must be binary or regression, {} is illegal".format(task_type)
                )

        # call the base
        super(NTasksNTowersSharedBottom, self).__init__(
            features,
            init_std,
            device,
            seed,
            tensorboard_path
        )
        
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_dnn = l2_reg_dnn
        self.init_std = init_std

        # For DNN block
        self.bottom_dnn_hidden_units = bottom_dnn_hidden_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.dnn_use_bn = dnn_use_bn

        self.task_types = task_types
        self.device = device
        self.gpus = gpus

        self.task_names = task_names

        self.bottom_dnn = DNN(
            self.input_dim,
            self.bottom_dnn_hidden_units,
            activation=self.dnn_activation,
            dropout_rate=self.dnn_dropout,
            use_bn=self.dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        if len(self.tower_dnn_hidden_units) > 0:
            # Another n blocks of MLP (DNN) for each tower
            self.tower_dnn = nn.ModuleList(
                [
                    DNN(
                        self.bottom_dnn_hidden_units[-1],
                        self.tower_dnn_hidden_units,
                        activation=self.dnn_activation,
                        dropout_rate=self.dnn_dropout,
                        use_bn=self.dnn_use_bn,
                        init_std=init_std,
                        device=device,
                    )
                    for _ in range(self.num_tasks)
                ]
            )

            # TODO (needs to do research on this)
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.tower_dnn.named_parameters(),
                )
            )

        # Note: the output for each layer is just a number
        # we do not to final tune at this moment
        self.tower_dnn_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    tower_dnn_hidden_units[-1]
                    if len(self.tower_dnn_hidden_units) > 0
                    else self.bottom_dnn_hidden_units[-1],
                    1,
                    bias=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

        # outs
        self.outs = nn.ModuleList([PredictionLayer(task) for task in task_types])
        
        # testing to show the two ouputs
        # self.output_one = PredictionLayer(task_types[0])
        # self.output_two = PredictionLayer(task_types[1])
        
        # need to undertand these regularization weight
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.bottom_dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.tower_dnn_final_layer.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )

        # attach to device tha we have
        self.to(device)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with
        # get  get_regularization_loss()

        if isinstance(weight_list, nn.parameter.Parameter):
            weight_list = [weight_list]

        # For generators, filters, and ParameterLists, convert
        # them to a list of tensors to avoid bugs.
        # e.g. can't pickle generator objects when we save the model
        else:
            weight_list = list(weight_list)

        self.regularization_weight.append((weight_list, l1, l2))

    def forward(self, data):
        """Forwards function to compute

        Args:
            # to be filled
        Returns:
            # to be filled
        """

        sparse_embedding_list, dense_value_list = self.inputs_from_feature_columns(data)

        dnn_input = combine_dnn_input(
            sparse_embedding_list=sparse_embedding_list,
            dense_value_list=dense_value_list,
        )

        # print(f"_ dnn_input: {dnn_input}")
        shared_bottom_output = self.bottom_dnn(dnn_input)
        # print(f" --- shared_bottom_output: {shared_bottom_output} ")

        # tower dnn (task-specific)
        # note: n towers have the same MLP architecture
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](shared_bottom_output)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)

            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](shared_bottom_output)

            output = self.outs[i](tower_dnn_logit)
            task_outs.append(output)

        # what is torch cat means here
        task_outs = torch.cat(task_outs, -1)
        
        return task_outs
