# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import torch
from typing import Optional
import torch.nn.functional as F

sys.path.insert(0, os.getenv("speakernet"))

import speakernet.utils.utils as utils
from nnet import (AttentiveStatisticsPooling, LDEPooling,
                  MarginSoftmaxLoss, MultiHeadAttentionPooling,
                  MultiResolutionMultiHeadAttentionPooling,
                  ReluBatchNormTdnnLayer, SEResNet, SoftmaxLoss,
                  StatisticsPooling, SpeakerNet,
                  for_extract_embedding)


class Encoder(SpeakerNet):
    """ A senet framework """

    def init(
        self,
        inputs_dim: int,
        num_targets: int,
        dropout: float = 0.0,
        training: bool = True,
        extracted_embedding: str = "near",
        resnet_params: dict = {},
        pooling: str = "statistics",
        pooling_params: dict = {},
        fc1: bool = False,
        fc1_params: dict = {},
        fc2_params: dict = {},
        margin_loss: bool = False,
        margin_loss_params: dict = {},
        use_step: bool = False,
        step_params: dict = {},
        adacos: bool = False,
        transfer_from: str = "softmax_loss",
        features: str = "fbank",
        emb_dim: Optional[int] = None,
    ):

        # Params.
        default_resnet_params = {
            "head_conv": True,
            "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
            "head_maxpool": False,
            "head_maxpool_params": {"kernel_size": 3, "stride": 1, "padding": 1},
            "block": "BasicBlock",
            "layers": [3, 4, 6, 3],
            "planes": [32, 64, 128, 256],  # a.k.a channels.
            "convXd": 2,
            "norm_layer_params": {"momentum": 0.5, "affine": True},
            "se_ratio": 16,
            "full_pre_activation": True,
            "zero_init_residual": False,
        }

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

        default_fc_params = {
            "nonlinearity": "relu",
            "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True},
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

        resnet_params = utils.assign_params_dict(default_resnet_params, resnet_params)
        pooling_params = utils.assign_params_dict(
            default_pooling_params, pooling_params
        )
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params
        )
        step_params = utils.assign_params_dict(default_step_params, step_params)

        # Var.
        self.extracted_embedding = extracted_embedding  # only near here.
        self.use_step = use_step
        self.step_params = step_params
        self.convXd = resnet_params["convXd"]

        # [batch, 1, feats-dim, frames] for 2d and  [batch, feats-dim, frames] for 1d.
        # Should keep the channel/plane is always in 1-dim of tensor (index-0 based).
        inplanes = 1 if self.convXd == 2 else inputs_dim
        self.resnet = SEResNet(inplanes, **resnet_params)

        # It is just equal to Ceil function.
        # 第 153 行将 channel 和 feat_dim 乘在了一起，因为statisticpoolinglayer只接受3维的tensor
        resnet_output_dim = (
            (inputs_dim + self.resnet.get_downsample_multiple() - 1)
            // self.resnet.get_downsample_multiple()
            * self.resnet.get_output_planes()
            if self.convXd == 2
            else self.resnet.get_output_planes()
        )

        # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.stats = LDEPooling(resnet_output_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(
                resnet_output_dim,
                hidden_size=pooling_params["hidden_size"],
                context=pooling_params["context"],
                stddev=stddev,
            )
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(
                resnet_output_dim, stddev=stddev, **pooling_params
            )
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(
                resnet_output_dim, **pooling_params
            )
        else:
            self.stats = StatisticsPooling(resnet_output_dim, stddev=stddev)

        self.fc1 = (
            ReluBatchNormTdnnLayer(
                self.stats.get_output_dim(), resnet_params["planes"][3], **fc1_params
            )
            if fc1
            else None
        )

        if fc1:
            fc2_in_dim = resnet_params["planes"][3]
        else:
            fc2_in_dim = self.stats.get_output_dim()

        # embedding dim: resnet_params["planes"][3]
        if emb_dim is None:
            emb_dim = resnet_params["planes"][3]
        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, emb_dim, **fc2_params)

        self.dropout = (
            torch.nn.Dropout2d(p=dropout) if dropout > 0 else None
        )

        # Do not need when extracting embedding.
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(
                    emb_dim, num_targets, **margin_loss_params
                )
            elif adacos:
                self.loss = AdaCos(emb_dim, num_targets)
            else:
                self.loss = SoftmaxLoss(emb_dim, num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["resnet", "stats", "fc1", "fc2"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight": "loss.weight"}

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = inputs
        # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        x = self.auto(self.aug_dropout, x)
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        x = (
            x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            if self.convXd == 2
            else x
        )
        # [256, 81, 300] -> [256, 2816, 38]
        x = self.stats(x)
        x = self.auto(self.fc1, x)
        x = self.fc2(x)
        x = self.auto(self.dropout, x)

        return x

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = (
            x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            if self.convXd == 2
            else x
        )
        x = self.stats(x)

        with torch.cuda.amp.autocast(enabled=False):
            if self.extracted_embedding == "far":
                assert self.fc1 is not None
                xvector = self.fc1.affine(x)
            elif self.extracted_embedding == "near_affine":
                x = self.auto(self.fc1, x)
                xvector = self.fc2.affine(x)
            elif self.extracted_embedding == "near":
                x = self.auto(self.fc1, x)
                xvector = self.fc2(x)
            else:
                raise TypeError(
                    "Expected far or near position, but got {}".format(
                        self.extracted_embedding
                    )
                )

        return xvector


# Test.
if __name__ == "__main__":
    # Let bach-size:128, fbank:40, frames:200.
    tensor = torch.randn(128, 40, 200)
    print("Test resnet2d ...")
    resnet2d = Encoder(40, 1211, resnet_params={"convXd": 2})
    print(resnet2d)
    print(resnet2d(tensor).shape)
    print("\n")
    print("Test resnet1d ...")
    resnet1d = Encoder(40, 1211, resnet_params={"convXd": 1})
    print(resnet1d)
    print(resnet1d(tensor).shape)

    print("Test done.")
