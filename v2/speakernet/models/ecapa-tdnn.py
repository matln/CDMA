""" Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))
sys.path.insert(0, os.path.dirname("/data/lijianchen/workspace/sre/speaker-net/speakernet"))

import speakernet.utils.utils as utils
from speakernet.models.nnet import (
    Conv1dBnReluLayer,
    SpeakerNet,
    MarginSoftmaxLoss,
    SoftmaxLoss,
    for_extract_embedding,
)


class Res2Conv1dReluBn(nn.Module):
    """
    Res2Conv1d + BatchNorm1d + ReLU

    in_channels == out_channels == channels
    """

    def __init__(self, channels, context=[0], bias=True, scale=4, tdnn_params={}):
        super().__init__()
        default_tdnn_params = {
            "nonlinearity": "relu",
            "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True},
        }

        tdnn_params = utils.assign_params_dict(default_tdnn_params, tdnn_params)

        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.blocks = nn.ModuleList(
            [
                Conv1dBnReluLayer(self.width, self.width, context, **tdnn_params, bias=bias)
                for i in range(self.nums)
            ]
        )

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.blocks[i](sp)
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class SE_Connect(nn.Module):
    def __init__(self, channels, s=4):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        assert channels // s == 128
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(nn.Module):
    """ SE-Res2Block.
    """

    def __init__(self, channels, context, scale, tdnn_layer_params={}):
        super().__init__()
        self.se_res2block = nn.Sequential(
            # Order: conv -> relu -> bn
            Conv1dBnReluLayer(channels, channels, context=[0], **tdnn_layer_params),
            Res2Conv1dReluBn(channels, context, scale=scale, tdnn_params=tdnn_layer_params,),
            # Order: conv -> relu -> bn
            Conv1dBnReluLayer(channels, channels, context=[0], **tdnn_layer_params),
            # SEBlock(channels, ratio=4),
            SE_Connect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class AttentiveStatsPool(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(AttentiveStatsPool, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim, kernel_size=1
            )  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim, kernel_size=1
            )  # equals W and b in the paper
        self.linear2 = nn.Conv1d(
            bottleneck_dim, in_dim, kernel_size=1
        )  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)


class Encoder(SpeakerNet):
    def init(
        self,
        inputs_dim,
        num_targets,
        channels=512,
        emb_dim=192,
        dropout=0.0,
        training=True,
        extracted_embedding="near",
        tdnn_layer_params={},
        layer5_params={},
        fc1=False,
        fc1_params={},
        fc2_params={},
        margin_loss=True,
        margin_loss_params={},
        pooling="ASTP",
        pooling_params={},
        use_step=True,
        step_params={},
        features: str = "fbank",
        norm_var: bool = False,
    ):
        default_tdnn_layer_params = {
            "nonlinearity": "relu",
            "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True},
        }

        default_layer5_params = {"nonlinearity": "relu", "bn": False}

        default_fc2_params = {"nonlinearity": "", "bn": True}

        default_pooling_params = {
            "num_head": 1,
            "hidden_size": 64,
            "share": True,
            "affine_layers": 1,
            "context": [0],
            "stddev": True,
            "temperature": False,
            "fixed": True,
        }

        default_margin_loss_params = {
            "method": "am",
            "m": 0.2,
            "feature_normalize": True,
            "s": 30,
            "mhe_loss": False,
            "mhe_w": 0.01,
        }

        default_step_params = {
            "T": None,
            "m": False,
            "lambda_0": 0,
            "lambda_b": 1000,
            "alpha": 5,
            "gamma": 1e-4,
            "s": False,
            "s_tuple": (30, 12),
            "s_list": None,
            "t": False,
            "t_tuple": (0.5, 1.2),
            "p": False,
            "p_tuple": (0.5, 0.1),
        }

        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        layer5_params = utils.assign_params_dict(default_layer5_params, layer5_params)
        layer5_params = utils.assign_params_dict(default_tdnn_layer_params, layer5_params)
        fc1_params = utils.assign_params_dict(default_tdnn_layer_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc2_params, fc2_params)
        fc2_params = utils.assign_params_dict(default_tdnn_layer_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params
        )
        step_params = utils.assign_params_dict(default_step_params, step_params)

        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding  # For extract.
        self.features = features
        self.norm_var = norm_var
        self.inputs_dim = inputs_dim
        self.margin_loss = margin_loss

        self.layer1 = Conv1dBnReluLayer(
            inputs_dim, channels, [-2, -1, 0, 1, 2], **tdnn_layer_params
        )
        # channels, kernel_size, stride, padding, dilation, scale
        self.layer2 = SE_Res2Block(channels, [-2, 0, 2], 8, tdnn_layer_params)
        self.layer3 = SE_Res2Block(channels, [-3, 0, 3], 8, tdnn_layer_params)
        self.layer4 = SE_Res2Block(channels, [-4, 0, 4], 8, tdnn_layer_params)

        cat_channels = channels * 3
        self.layer5 = Conv1dBnReluLayer(cat_channels, cat_channels, [0], **layer5_params)

        if pooling == "ASTP":
            self.pooling = AttentiveStatsPool(cat_channels, 128, global_context_att=True)
            self.bn_pool = nn.BatchNorm1d(cat_channels * 2)
            self.fc1 = (
                Conv1dBnReluLayer(cat_channels * 2, emb_dim, **fc1_params) if fc1 else None
            )
        else:
            raise ValueError
            # self.pooling = StatisticsPooling(cat_channels, stddev=True)

        if fc1:
            fc2_in_dim = emb_dim
        else:
            fc2_in_dim = cat_channels * 2
        self.fc2 = Conv1dBnReluLayer(fc2_in_dim, emb_dim, **fc2_params)
        self.dropout = torch.nn.Dropout2d(p=dropout) if dropout > 0 else None

        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(emb_dim, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(emb_dim, num_targets)

    @utils.for_device_free
    def forward(self, inputs):
        """
        inputs: [batch, features-dim, frames-lens]
        """
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # 在 channel 维连接
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.layer5(out)
        out = self.bn_pool(self.pooling(out).unsqueeze(-1))
        out = self.auto(self.fc1, out)
        out = self.fc2(out)
        out = self.auto(self.dropout, out)
        return out

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # 在 channel 维连接
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.layer5(out)
        out = self.bn_pool(self.pooling(out))

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(out)
        elif self.extracted_embedding == "near_affine":
            out = self.auto(self.fc1, out)
            xvector = self.fc2.affine(out)
        elif self.extracted_embedding == "near":
            out = self.auto(self.fc1, out)
            xvector = self.fc2(out)

        return xvector


if __name__ == "__main__":
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(128, 80, 200)
    model = Encoder(inputs_dim=80, num_targets=5994, channels=512, emb_dim=192)
    # out = model(x)
    print(model)
    # print(out.shape)    # should be [2, 192]

    import numpy as np

    print(np.sum([p.numel() for p in model.parameters()]).item())
