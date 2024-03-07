# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import sys
import os
import torch
import torch.nn.functional as F

subtools = '/data/lijianchen/workspace/sre/subtools'
# subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))

from libs.nnet import *
import libs.support.utils as utils


class Xvector(TopVirtualNnet):
    """ A standard x-vector framework """

    def init(self, inputs_dim, num_targets, aug_dropout=0.2, training=True,
             extracted_embedding="far", tdnn_layer_params={}):

        default_tdnn_layer_params = {
            "nonlinearity": 'relu',
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True}
        }
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)

        # Var
        self.extracted_embedding = extracted_embedding

        # Nnet
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim, 512, [-2, -1, 0, 1, 2], **tdnn_layer_params)
        self.tdnn2 = ReluBatchNormTdnnLayer(512, 512, [-2, 0, 2], **tdnn_layer_params)
        self.tdnn3 = ReluBatchNormTdnnLayer(512, 512, [-3, 0, 3], **tdnn_layer_params)
        self.tdnn4 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params)
        self.tdnn5 = ReluBatchNormTdnnLayer(512, 1500, **tdnn_layer_params)
        self.stats = StatisticsPooling(1500, stddev=True)
        self.tdnn6 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), 512, **tdnn_layer_params)
        self.tdnn7 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params)

        # Do not need when extracting embedding.
        if training:
            self.loss = SoftmaxLoss(512, num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["tdnn1", "tdnn2", "tdnn3",
                                   "tdnn4", "tdnn5", "stats", "tdnn6", "tdnn7"]

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        x = inputs
        # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        x = self.auto(self.aug_dropout, x)

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats(x)
        x = self.tdnn6(x)
        outputs = self.tdnn7(x)

        return outputs

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        return self.loss(inputs, targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats(x)

        if self.extracted_embedding == "far":
            xvector = self.tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.tdnn6(x)
            xvector = self.tdnn7.affine(x)

        return xvector


# Test.
if __name__ == "__main__":
    model = Xvector(inputs_dim=26, num_targets=1211, training=False)
    print(model)
    # print(out.shape)    # should be [2, 192]

    import numpy as np
    print(np.sum([p.numel() for p in model.parameters()]).item())

    # from collections import OrderedDict
    # import numpy as np

    # model = Xvector(26, 1211)
    # fast_weights = OrderedDict(model.named_parameters())
    # inputs = torch.Tensor(2, 26, 200)
    # inputs2 = torch.Tensor(2, 26, 200)
    # targets = np.array([0, 1])
    # targets2 = np.array([0, 1])

    # weights = OrderedDict(model.named_parameters())
    # out = model.forward(inputs)
    # loss = model.get_loss(out, targets)
    # gradients = torch.autograd.grad(loss, weights.values(), create_graph=False)

    # out2 = model.forward(inputs2)
    # loss2 = model.get_loss(out2, targets2)
    # loss2.backward()

    # # weights = OrderedDict(
    # #     (name, param - grad)
    # #     for ((name, param), grad) in zip(weights.items(), gradients))

    # #         p.grad.detach().mul_(clip_coef.to(p.grad.device))
    # weights = OrderedDict(model.named_parameters())
    # grad = torch.ones((512, 26, 5))
    # # for ((name, param), grad) in zip(weights.items(), gradients):
    # for (name, param) in weights.items():
    #     print(name)
    #     # print(param.grad)
    #     # print(param.grad.detach().add_(3 / 4 * grad.to(param.grad.device)))
    # # # One step update
    # # weights = OrderedDict(
    # #     (name, param - 4 * alpha * grad)
    # #     for ((name, param), grad) in zip(weights.items(), gradients))
