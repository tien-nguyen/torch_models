import torch.nn as nn


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(name: str, hidden_size: int):
    """Construct activation layers
    Args:
        name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation // Data Adaptive Activation Function
    Return:
        act_layer: activation_layer

    """
    if isinstance(name, str):
        name = name.lower()

        if name == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif name == 'linear':
            act_layer = Identity()
        elif name == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif name == "dice":
            raise NotImplementedError
        elif name == "prelu":
            act_layer = nn.PReLU()
    elif issubclass(name, nn.Module):
        act_layer = name()
    else:
        raise NotImplementedError

    return act_layer
