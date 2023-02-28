"""
Author:
    Tien T. Nguyen
    nguyen.ttq.tien@gmail.com
"""
import time
from typing import Callable, List, Mapping, Union

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
                                           create_embedding_matrix)
from jup.recsys_models.core.utils import slice_arrays
from jup.recsys_models.features import (DenseFeature, SparseFeature,
                                        get_dense_feature, get_sparse_feature)
from jup.recsys_models.inputs import compute_input_dim


class BaseModel(nn.Module):
    """ Base class for all models in this repo.
    
    All of the models should subclass this class.
    
    This class contains some common utils for all models to use such as:
        * embedding dict creation for sparse feature
        * train step
        * eval step
        * predict step
        
    It does not, however, contain the forward method.
    """
    
    def __init__(
        self, 
        features: List[Union[SparseFeature, DenseFeature]],
        embedding_init_std: float = 0.0001, 
        device: str = "cpu",
        seed=1024,
        tensorboard_path=None,
        **kwargs
    ):
        """
        Args:
            features: an list of features used by the model
            emdding_init_std: float, to use as the initialize std of embedding vector
            device: cpu or gpu
            seed: integer, to use as random seed.
    
        """
        super(BaseModel, self).__init__()
        
        self.device = device
        self.seed = seed  # not sure what is this for.
        self.features = features
        self._dense_features = None
        self._sparse_features = None
        
        # build feature column index
        self.feature_col_index = build_input_feature_column_index(
            self.features
        )
        
        # Embedding for Sparse features
        self.embedding_init_std = embedding_init_std
        
        # building the embedding dict for all the sparse features
        self.embedding_dict = create_embedding_matrix(
            self.sparse_features, self.embedding_init_std,
            device=self.device
        )
        
        # regularization_weight - @ TODO: will need to understand this.
        self.regularization_weight = []
        
        # attach to the device that we have
        self.to(self.device)
        
        # tensorboard
        self.tensorboard_path = tensorboard_path
        self.tensorboard_writer = None
        if self.tensorboard_path:
            self.tensorboard_writer = SummaryWriter(self.tensorboard_path)
            
        # input_dim
        self.input_dim = compute_input_dim(features)
    
    @property
    def dense_features(self):
        if not self._dense_features:
            self._dense_features =  get_dense_feature(
                features=self.features
            )
        return self._dense_features
        
    @property
    def sparse_features(self):
        if not self._sparse_features:
            self._sparse_features = get_sparse_feature(
                features=self.features
            )
            
        return self._sparse_features

    def inputs_from_feature_columns(
        self, data
    ):
        """Extracts features from data

        Args:
            data: ## to be filled
            feature_colums: to be filed

        Returns: # need to annotate the return here

        Notes:
            We will support dense feature by default
            Not support VarLen Sparse feature at this time.

        """

        sparse_embedding_list = [
            self.embedding_dict[feature.embedding_name](
                data[
                    :,
                    self.feature_col_index[feature.name][0] : self.feature_col_index[
                        feature.name
                    ][1],
                ].long()
            )
            for feature in self.sparse_features
        ]

        # for dense features, we do not need to extract from the embedding
        dense_feature_list = [
            data[
                :,
                self.feature_col_index[feature.name][0] : self.feature_col_index[
                    feature.name
                ][1],
            ]
            for feature in self.dense_features
        ]

        return sparse_embedding_list, dense_feature_list

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

    def _get_loss_func(self, loss) -> Callable:
        if loss == "binary_cross_entropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

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

    def evaluate(self, x, y, batch_size=256):
        """
        # TODO: write more explanation here
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

    def predict(self, x, batch_size=256):
        """
        # TODO: writer more explanation here
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

                        if self.tensorboard_writer:
                            self.tensorboard_writer.add_graph(model, x)
                            self.tensorboard_writer.flush()

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
            
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar("training_loss", epoch_logs['loss'], global_step=epoch)
                
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
        
        if self.tensorboard_writer:
            self.tensorboard_writer.flush()