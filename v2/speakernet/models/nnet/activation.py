# -*- coding:utf-8 -*-
"""
Copyright 2019 Snowdar
          2022 Jianchen Li
"""

import torch
import numpy as np
import torch.nn.functional as F


class Mish(torch.nn.Module):
    """A changed ReLU. 
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, inputs):
        return inputs * torch.tanh(F.softplus(inputs))


# Wrapper ✿


def Nonlinearity(nonlinearity="relu", inplace=True, negative_slope=0.01):
    """A wrapper for activation.
    """
    if nonlinearity == "relu":
        activation = torch.nn.ReLU(inplace=inplace)
    elif nonlinearity == "leaky_relu":
        activation = torch.nn.LeakyReLU(inplace=inplace, negative_slope=negative_slope)
    elif nonlinearity == "selu":
        activation = torch.nn.SELU(inplace=inplace)
    elif nonlinearity == "mish":
        activation = Mish()
    elif nonlinearity == "tanh":
        activation = torch.nn.Tanh()
    elif nonlinearity == "" or nonlinearity is None or nonlinearity is False:
        activation = None
    else:
        raise ValueError("Do not support {0} nonlinearity now.".format(nonlinearity))

    return activation
