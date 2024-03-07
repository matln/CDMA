# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import os
import sys
import math
import torch
import torch.nn.functional as F

subtools = '/data/lijianchen/workspace/sre/subtools'
# subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))

import libs.support.utils as utils
from libs.nnet import *


class Xvector(TopVirtualNnet):
    """ A composite x-vector framework
    在 extended-xvector 基础上多了 SEBlock、mixup、specaugment 等等
    """

    # Base parameters - components - loss - training strategy.
    def init(self, inputs_dim, num_targets, extend=False, skip_connection=False,
             mixup=False, mixup_alpha=1.0,
             specaugment=False, specaugment_params={},
             aug_dropout=0., context_dropout=0., hidden_dropout=0., dropout_params={},
             SE=False, se_ratio=4,
             tdnn_layer_params={}, emb_dim=512,
             tdnn6=True, tdnn7_params={},
             pooling="statistics", pooling_params={},
             margin_loss=False, margin_loss_params={},
             use_step=False, step_params={},
             transfer_from="softmax_loss",
             training=True, extracted_embedding="far"):

        # Params.
        default_dropout_params = {
            "type": "default",  # default | random
            "start_p": 0.,
            "dim": 2,
            "method": "uniform",  # uniform | normals
            "continuous": False,
            "inplace": True
        }

        default_tdnn_layer_params = {
            "affine_type": 'tdnn-affine',
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False, "bn": True, "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}
        }

        default_pooling_params = {
            "num_nodes": 1500,
            "num_head": 1,
            "share": True,
            "affine_layers": 1,
            "hidden_size": 64,
            "context": [0],
            "stddev": True,
            "temperature": False,
            "fixed": True,
            "stddev": True
        }

        default_margin_loss_params = {
            "method": "am", "m": 0.2,
            "feature_normalize": True, "s": 30,
            "double": False,
            "mhe_loss": False, "mhe_w": 0.01,
            "inter_loss": 0.,
            "ring_loss": 0.,
            "curricular": False
        }

        default_step_params = {
            "T": None,
            "m": False, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
            "s": False, "s_tuple": (30, 12), "s_list": None,
            "t": False, "t_tuple": (0.5, 1.2),
            "p": False, "p_tuple": (0.5, 0.1)
        }

        dropout_params = utils.assign_params_dict(default_dropout_params, dropout_params)
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)
        # If param is not be specified, default it w.r.t tdnn_layer_params.
        tdnn7_params = utils.assign_params_dict(tdnn_layer_params, tdnn7_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        # Var.
        self.skip_connection = skip_connection
        self.use_step = use_step
        self.step_params = step_params

        self.extracted_embedding = extracted_embedding  # For extract.

        # Nnet.
        # Head
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None
        # nnet.dropout.SpecAugment
        self.specaugment = SpecAugment(**specaugment_params) if specaugment else None
        self.aug_dropout = get_dropout_from_wrapper(aug_dropout, dropout_params)
        self.context_dropout = ContextDropout(p=context_dropout) if context_dropout > 0 else None
        self.hidden_dropout = get_dropout_from_wrapper(hidden_dropout, dropout_params)

        # Frame level
        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim, 512, [-2, -1, 0, 1, 2], **tdnn_layer_params)
        self.se1 = SEBlock(512, ratio=se_ratio) if SE else None
        self.ex_tdnn1 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params) if extend else None
        self.tdnn2 = ReluBatchNormTdnnLayer(512, 512, [-2, 0, 2], **tdnn_layer_params)
        self.se2 = SEBlock(512, ratio=se_ratio) if SE else None
        self.ex_tdnn2 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params) if extend else None
        self.tdnn3 = ReluBatchNormTdnnLayer(512, 512, [-3, 0, 3], **tdnn_layer_params)
        self.se3 = SEBlock(512, ratio=se_ratio) if SE else None
        self.ex_tdnn3 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params) if extend else None
        self.ex_tdnn4 = ReluBatchNormTdnnLayer(512, 512, [-4, 0, 4], **tdnn_layer_params) if extend else None
        self.se4 = SEBlock(512, ratio=se_ratio) if SE and extend else None
        self.ex_tdnn5 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params) if extend else None
        self.tdnn4 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params)

        num_nodes = pooling_params.pop("num_nodes")

        self.tdnn5 = ReluBatchNormTdnnLayer(512, num_nodes, **tdnn_layer_params)

        # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.stats = LDEPooling(num_nodes, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(num_nodes, affine_layers=pooling_params["affine_layers"],
                                                    hidden_size=pooling_params["hidden_size"],
                                                    context=pooling_params["context"], stddev=stddev)
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(num_nodes, stddev=stddev, **pooling_params)
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(num_nodes, **pooling_params)
        elif pooling == "xi-postmean-softplus2":
            self.stats = xivec_stdinit_softplus2_prec_pooling(num_nodes, hidden_size=pooling_params["hidden_size"], stddev=False)
        elif pooling == "xi-postdist-softplus2":
            self.stats = xivec_stdinit_softplus2_prec_pooling(num_nodes, hidden_size=pooling_params["hidden_size"], stddev=True)
        else:
            self.stats = StatisticsPooling(num_nodes, stddev=stddev)

        stats_dim = self.stats.get_output_dim()

        # Segment level
        if tdnn6:
            self.tdnn6 = ReluBatchNormTdnnLayer(stats_dim, 512, **tdnn_layer_params)
            tdnn7_dim = 512
        else:
            self.tdnn6 = None
            tdnn7_dim = stats_dim

        if tdnn7_params["nonlinearity"] == "default":
            tdnn7_params["nonlinearity"] = tdnn_layer_params["nonlinearity"]

        self.tdnn7 = ReluBatchNormTdnnLayer(tdnn7_dim, emb_dim, **tdnn7_params)

        # Loss
        # Do not need when extracting embedding.
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(emb_dim, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(emb_dim, num_targets)

            self.wrapper_loss = MixupLoss(self.loss, self.mixup) if mixup else None

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["tdnn1", "tdnn2", "tdnn3", "tdnn4", "tdnn5", "stats", "tdnn6", "tdnn7",
                                   "ex_tdnn1", "ex_tdnn2", "ex_tdnn3", "ex_tdnn4", "ex_tdnn5",
                                   "se1", "se2", "se3", "se4", "loss"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {
                    "loss.affine.weight": "loss.weight"}

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """

        x = inputs

        x = self.auto(self.mixup, x)
        x = self.auto(self.specaugment, x)
        x = self.auto(self.aug_dropout, x)
        x = self.auto(self.context_dropout, x)

        x = self.tdnn1(x)
        if self.skip_connection:
            identity = x
        x = self.auto(self.se1, x)
        x = self.auto(self.ex_tdnn1, x)
        x = self.tdnn2(x)
        x = self.auto(self.se2, x)
        x = self.auto(self.ex_tdnn2, x)
        x = self.tdnn3(x)
        x = self.auto(self.se3, x)
        x = self.auto(self.ex_tdnn3, x)
        x = self.auto(self.ex_tdnn4, x)
        x = self.auto(self.se4, x)
        x = self.auto(self.ex_tdnn5, x)
        x = self.tdnn4(x)
        if self.skip_connection:
            x = x + identity
        x = self.tdnn5(x)
        x = self.stats(x)
        x = self.auto(self.tdnn6, x)
        x = self.tdnn7(x)
        x = self.auto(self.hidden_dropout, x)
        return x

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss(inputs, targets)
        else:
            return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss().
        @return: return accuracy
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_accuracy(targets)
        else:
            return self.loss.get_accuracy(targets)

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs

        x = self.tdnn1(x)
        if self.skip_connection:
            identity = x
        x = self.auto(self.se1, x)
        x = self.auto(self.ex_tdnn1, x)
        x = self.tdnn2(x)
        x = self.auto(self.se2, x)
        x = self.auto(self.ex_tdnn2, x)
        x = self.tdnn3(x)
        x = self.auto(self.se3, x)
        x = self.auto(self.ex_tdnn3, x)
        x = self.auto(self.ex_tdnn4, x)
        x = self.auto(self.se4, x)
        x = self.auto(self.ex_tdnn5, x)
        x = self.tdnn4(x)
        if self.skip_connection:
            x = x + identity
        x = self.tdnn5(x)
        x = self.stats(x)

        if self.extracted_embedding == "far":
            assert self.tdnn6 is not None
            xvector = self.tdnn6.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.tdnn6, x)
            xvector = self.tdnn7.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.tdnn6, x)
            xvector = self.tdnn7(x)

        return xvector


if __name__ == "__main__":

    model_params = {
        "extend": True,
        "SE": False,
        "se_ratio": 4,
        "training": True, "extracted_embedding": "far",

        "tdnn_layer_params": {"affine_type": 'conv1d-dilation',
                              "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
                              "bn-relu": False,
                              "bn": True,
                              "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

        "pooling": "statistics",  # statistics, lde, attentive, multi-head, multi-resolution
        "pooling_params": {"num_nodes": 1500,
                           "num_head": 1,
                           "share": True,
                           "affine_layers": 1,
                           "hidden_size": 64,
                           "context": [0],
                           "temperature": False,
                           "fixed": True},
        "tdnn6": True,
        "tdnn7_params": {"nonlinearity": "", "bn": True},

        "margin_loss": True,
        "margin_loss_params": {"method": "am", "m": 0.2, "feature_normalize": True,
                               "s": 30, "mhe_loss": False, "mhe_w": 0.01},
        "use_step": True,
        "step_params": {"T": None,
                        "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
                        "s": False, "s_tuple": (30, 12), "s_list": None,
                        "t": False, "t_tuple": (0.5, 1.2),
                        "p": False, "p_tuple": (0.5, 0.1)}
    }

    x = torch.zeros(2, 26, 200)
    model = Xvector(inputs_dim=26, num_targets=1211, **model_params)
    out = model(x)
    print(model)
    # # print(out.shape)    # should be [2, 192]

    import numpy as np
    print(np.sum([p.numel() for p in model.tdnn3.affine.parameters()]).item())
    print(np.sum([p.numel() for p in model.parameters()]).item())


