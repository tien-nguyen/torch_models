import torch
import torch.nn as nn


class PredictionLayer(nn.Module):
    """The final layer to generate the output

    Args:
        task: str
            binary for binary logloss
            regression for regression loss

        use_bias: bool
            whether add bias term or not
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):

        if task not in ['binary', 'multiclass', "regression"]:
            raise ValueError(
                "task must be in binary, multiclass or regression")

        super(PredictionLayer, self).__init__()

        self.use_bias = use_bias
        self.task = task

        if self.use_bias:
            # TODO (add more explanation)
            self.bias = nn.Parameter(torch.zeros((1, )))

    def forward(self, inputs):
        outputs = inputs

        if self.use_bias:
            outputs += self.bias

        if self.task == "binary":
            outputs = torch.sigmoid(outputs)

        return outputs
