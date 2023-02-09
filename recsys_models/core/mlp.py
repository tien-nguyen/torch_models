import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jup.recsys_models.core.activation_layer import activation_layer


class DNN(nn.Module):
    """The Multi Layer Perception

    Input shape:
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input
        with shape ``(batch_size, input_dim)``.

    Output shape:
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.

    Args:
        inputs_dim: input feature dimension

        hidden_units: list of positive integer, the layer number and units in each layer

        activation: activation function to use

        l2_reg: float between 0 and 1.
            L2 regularizer strength applied to the kernel weights matrix.

        dropout_rate:
            float in [0,1). Fraction of the units to drop. We cannot drop all of it

        use_bn: bool
            whether use BatchNormalization before activation or not

        seed: A python integer to use as random seed.
    """

    def __init__(self, input_dims, hidden_units, activation='relu', l2_reg=0,
                 dropout_rate=0, use_bn=False, init_std=0.0001, dice_dim=3, seed=1024,
                 device='cpu'):

        super(DNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        if len(hidden_units) == 0:
            raise ValueError("hidden units is empty!")

        # this is the DNN layer, everything is hidden
        hidden_units = [input_dims] + list(hidden_units)

        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # can be indexed like a regular Python list, but modules it contains
        # are properly registerd and will be visibile by all Module methods

        # say we have 3 inputs, and 2 output,
        # thenw we will declare this as nn.Linear(3, 2)
        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_units[i], hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ]
        )

        # note, we just dont do the Batch Norm for the first and the last
        # nodes
        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(hidden_units[i + 1])
                    for i in range(len(hidden_units) - 1)
                ]
            )

        self.activation_layes = nn.ModuleList(
            [
                activation_layer(activation,
                                 hidden_units[i + 1],
                                 dice_dim)
                for i in range(len(hidden_units) - 1)
            ]
        )

        for name, tensor in self.linears.named_parameters():
            # TODO: we may have a different way to intialize the tensor
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            # fc = fully connected
            fc = self.linears[i](deep_input)

            # batch normalized
            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layes[i](fc)
            fc = self.dropout(fc)

            deep_input = fc

        return deep_input
