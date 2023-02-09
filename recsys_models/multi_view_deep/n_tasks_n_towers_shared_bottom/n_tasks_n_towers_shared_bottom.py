import os
import torch
from torch import nn
from torch.nn import functional as F
from jup.recsys_models.features import compute_input_dim
from jup.recsys_models.core.mlp import DNN
from jup.recsys_models.core.prediction_layer import PredictionLayer
from jup.recsys_models.core.inputs import create_embedding_matrix
from jup.recsys_models.core.inputs import build_input_feature_column_index
from jup.recsys_models.core.inputs import combine_dnn_input
from jup.recsys_models.core.utils import slice_arrays

from typing import List
from typing import Union
from jup.recsys_models.features import DenseFeature
from jup.recsys_models.features import SparseFeature
from jup.recsys_models.features import get_sparse_feature
from jup.recsys_models.features import get_dense_feature

import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

import numpy as np
import time

from tqdm import tqdm
from typing import Mapping
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

class NTasksNTowersSharedBottom(nn.Module):
    """https://arxiv.org/pdf/1706.05098.pdf

    Creates a network of two towers where its shared the same bottom DNN

    Args
        dnn_feature_columns: An iterable containing all the features used by deep part of the model.
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
        dnn_feature_columns,
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
    ):
        # some checks
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ["binary", "regression"]:
                raise ValueError(
                    "task must be binary or regression, {} is illegal".format(task_type)
                )

        super(NTasksNTowersSharedBottom, self).__init__()
        # set values
        self.dnn_feature_columns = dnn_feature_columns

        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_dnn = l2_reg_dnn
        self.init_std = init_std
        self.seed = seed  # not sure if this is for.

        # build up the embeddding dict for sparse features
        # consider move to a Base one
        self.embedding_dict = create_embedding_matrix(
            self.dnn_feature_columns, self.init_std
        )

        # also consider move this to a base class
        self.feature_col_index = build_input_feature_column_index(
            self.dnn_feature_columns
        )

        # For DNN block
        self.bottom_dnn_hidden_units = bottom_dnn_hidden_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.dnn_use_bn = dnn_use_bn

        self.task_types = task_types
        self.device = device
        self.gpus = gpus

        self.task_names = task_names
        self.input_dim = compute_input_dim(dnn_feature_columns)

        # regularization_weight
        self.regularization_weight = []

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

    # maybe move this to a common place later
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

    # maybe move this to a common place later
    def input_from_feature_columns(
        self, data, features: List[Union[DenseFeature, SparseFeature]]
    ):
        """Extracts features from data

        Args:
            data: ## to be filled
            feature_colums: to be filed

        Returns:

        Notes:
            We will support dense feature by default
            Not support VarLen Sparse feature at this time.

        """
        sparse_features = get_sparse_feature(features)

        # print(" in input from feature columsn ")
        # print(sparse_features)

        dense_features = get_dense_feature(features)
        # print(dense_features)

        # print("-----")

        # take everything, from column to that column
        # feature_index is just a location of the feature in the input vector

        # print("before calling to make sparse embedding list")
        # print(list(self.embedding_dict.keys()))

        # sparse_embedding_list = []

        # for feature in sparse_features:
        #     print(f"processing for {feature.name} with embedding name: {feature.embedding_name}")

        #     _from = self.feature_col_index[feature.name][0]
        #     _to = self.feature_col_index[feature.name][1]

        #     print(f"from {_from} to {_to}")
        #     _data = data[:, _from: _to]
        #     print(_data)

        #     embedding = self.embedding_dict[feature.embedding_name](data[:, _from: _to].long())

        #     print(" retrieved embedding: ")
        #     print(embedding)

        sparse_embedding_list = [
            self.embedding_dict[feature.embedding_name](
                data[
                    :,
                    self.feature_col_index[feature.name][0] : self.feature_col_index[
                        feature.name
                    ][1],
                ].long()
            )
            for feature in sparse_features
        ]

        # print(" --- sparse embedding list ---")
        # print(sparse_embedding_list)

        # for dense features, we do not need to extract from the embedding
        dense_feature_list = [
            data[
                :,
                self.feature_col_index[feature.name][0] : self.feature_col_index[
                    feature.name
                ][1],
            ]
            for feature in dense_features
        ]

        # print("--- dense_feature_list ---")
        # print(dense_feature_list)

        return sparse_embedding_list, dense_feature_list

    # consider move this to a common place
    def set_optimizer(self, optimizer: str) -> None:
        """Sets optimizer for the model

        Args:
             optimizer: str see https://pytorch.org/docs/stable/optim.html

        """
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                self.optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                self.optim = torch.optim.Adam(
                    self.parameters()
                )  # 0.001 <- what does it mean here?
            elif optimizer == "adagrad":
                self.optim = torch.optim.Adagrad(
                    self.parameters()
                )  # 0.01 <- what does it mean here?
            elif optimizer == "rmsprop":
                self.optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            self.optim = optimizer

    # consider move this to a common place
    def set_loss_function(self, loss: Union[List, str]) -> None:
        """Sets the loss function
        See: https://pytorch.org/docs/stable/nn.functional.html#loss-functions
        """
        if isinstance(loss, str):
            self.loss_func = self._get_loss_func(loss)
        elif isinstance(loss, list):
            self.loss_func = [self._get_loss_func(loss_single) for loss_single in loss]
        else:
            self.loss_func = loss

    # consider move this to a common place (base class)
    def _get_loss_func(self, loss):
        if loss == "binary_cross_entropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    # consider move this to a common place
    def set_metrics(self, metrics: List) -> None:
        """Sets a list of metrics used to evaluated by the model during training and testing"""
        self.metric_names = ["loss"]
        self.metrics = {}
        if metrics:
            for metric in metrics:
                if metric in ["binary_cross_entropy", "logloss"]:
                    self.metrics[metric] = log_loss  # why dont we use BCELoss ?
                else:
                    raise NotImplementedError

    # consider move this to a common place
    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    # consider to move this to a common place.
    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    # consider to move this to a common place
    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_col_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size
        )

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    # consider move this to a common place
    def fit(
        self,
        data: Mapping[str, pd.Series],
        labels,
        batch_size=256,
        epochs=5,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    ):
        """Trains a model with input data and labels

        Args:
            data:
                a numpy array of training data if it is a single input
                a list of numpy arrays  if model has multiple inputs

            labels:
                a numpy array of target (label) if the model has a single output
                a list of numpy array (if the model has multiple outputs)

            batch_size:
                number of samples per gradient update.

            epoch:
                number of epochs to train models

            verbose:
                0, 1, or 2. Verbosity mode.
                    0 = silent, 1 = progress bar, 2 = one line per epoch.

        """

        """
            Note: data can be a dictionary 
                where key is str
                value is pandas pandas.Series
                
            So we need to convert this dictionary to a list first
            the order of the feature based on the self.feature_col_index
        """
        
        # for Tensorflow Board
        # hard code for now
        # the path should be a variable to pass in
        writer = SummaryWriter('jup/recsys_models/multi_view_deep/two_task_two_tower_shared_bottom/runs/')
        input_data = [data[feature] for feature in self.feature_col_index]

        # we always do the validation while training model
        do_validation = True
        # not sure what it means by
        # hasattr(x[0], 'shape')

        # need to annotate this
        split_at = int(len(input_data[0]) * (1.0 - validation_split))
        train_data, val_data = (
            slice_arrays(input_data, 0, split_at),
            slice_arrays(input_data, split_at),
        )

        train_label, val_label = (
            slice_arrays(labels, 0, split_at),
            slice_arrays(labels, split_at),
        )

        # TODO (@tien): understand why we need this function
        for i in range(len(train_data)):
            if len(train_data[i].shape) == 1:
                train_data[i] = np.expand_dims(train_data[i], axis=1)

        # read this to understand
        # what it means by concatenate train_data, axis = -1
        # https://www.sharpsightlabs.com/blog/numpy-concatenate/
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(train_data, axis=-1)),
            torch.from_numpy(train_label),
        )

        # set model to train
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size
        )

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print(
            "Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
                len(train_tensor_data), len(val_data), steps_per_epoch
            )
        )

        train_result = {}
        epoch_logs = {}

        # To continue
        for epoch in range(epochs):
            loss_epoch = 0
            total_loss_epoch = 0
            start_time = time.time()

            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()


                        writer.add_graph(model, x)
                        writer.flush()

                        # https://pytorch.org/docs/stable/generated/torch.squeeze.html
                        y_pred = model(x).squeeze()
                        # forgot why we need to do this?
                        optim.zero_grad()

                        # so each of the head will have a different loss function
                        if isinstance(loss_func, list):
                            assert (
                                len(loss_func) == self.num_tasks
                            ), "the length of loss_func should be equal with 'self.num_tasks"

                            loss = sum(
                                [
                                    loss_func[i](y_pred[:, i], y[:, i], reduction="sum")
                                    for i in range(self.num_tasks)
                                ]
                            )
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction="sum")

                        reg_loss = self.get_regularization_loss()

                        total_loss = (
                            loss + reg_loss
                        )  # there is a + for self.aux_los, not sure what this is

                        # not exactly sure what it means by item here
                        # need to understand these things
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(
                                    metric_fun(
                                        y.cpu().data.numpy(),
                                        y_pred.cpu().data.numpy().astype("float64"),
                                    )
                                )

            except KeyboardInterrupt:
                # t.close()
                raise

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            
            writer.add_scalar("training_loss", epoch_logs['loss'], global_step=epoch)
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_data, val_label, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            
            
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print("Epoch {0}/{1}".format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"]
                )

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += (
                            " - "
                            + "val_"
                            + name
                            + ": {0: .4f}".format(epoch_logs["val_" + name])
                        )
                print(eval_str)
        writer.flush()

    def forward(self, data):
        """Forwards function to compute

        Args:
            # to be filled
        Returns:
            # to be filled
        """

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            data, self.dnn_feature_columns
        )

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
            # if i == 0:
            #     output = self.output_one(tower_dnn_logit)
            # else:
            #     output = self.output_two(tower_dnn_logit)
                
            task_outs.append(output)

        # what is torch cat means here
        task_outs = torch.cat(task_outs, -1)
        
        # output
        # print("task outs")
        # print(task_outs)
        return task_outs
