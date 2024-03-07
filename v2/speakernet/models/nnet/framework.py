# -*- coding:utf-8 -*-
"""
Copyright 2019 Snowdar
          2022 Jianchen Li
"""

import math
import torch
from typing import List, Dict

import speakernet.utils.utils as utils


def for_extract_embedding(maxChunk=10000):
    """
    A decorator for extract_embedding class-function to wrap some common process codes like Kaldi's x-vector extractor.
    Used in TopVirtualNnet.
    """

    def wrapper(function):
        def _wrapper(self, input):
            """
            @input: a 3-dimensional tensor [*, feature-dim, frames] or 2-dimensional tensor [*, feature-dim]
            @return: an 1-dimensional vector
            """
            train_status = self.training
            self.eval()

            with torch.no_grad():
                input = utils.to_device(self, input)
                if len(input.shape) == 3:
                    num_frames = input.shape[2]
                    num_split = (num_frames + maxChunk - 1) // maxChunk
                    # 最后一个chunk size会超过maxChunk吗？会
                    # 令S=num_frames/num_split, S'=num_frames//num_split, N=num_split
                    # 则(S-1)(N-1)<S'(N-1)<=S(N-1)
                    #   SN-S(N-1)<=SN-S'(N-1)<SN-(S-1)(N-1)
                    #   S<=SN-S'(N-1)<S+N-1
                    # 假如num_frames=999, maxChunk=100, 则num_split=10，S'=99，最后一个chunk为108
                    # 所以改为向上取整
                    # split_size = num_frames // num_split
                    split_size = (num_frames - 1) // num_split + 1

                    offset = 0
                    embedding_stats = 0.0
                    for i in range(0, num_split - 1):
                        # [1, emd-dim, 1]
                        this_embedding = function(self, input[:, :, offset : offset + split_size])
                        offset += split_size
                        embedding_stats += split_size * this_embedding

                    last_embedding = function(self, input[:, :, offset:])

                    embedding = (
                        embedding_stats + (num_frames - offset) * last_embedding
                    ) / num_frames
                else:
                    embedding = function(self, input)

                if train_status:
                    self.train()

                return torch.squeeze(embedding).cpu()

        return _wrapper

    return wrapper


# Relation: activation -> components -> loss -> framework

# Framework
class SpeakerNet(torch.nn.Module):
    """This is a nnet framework at top level and it is applied to the pipline scripts.
    And you should implement four functions after inheriting this object.

    @init(): just like pytorch needed. Note there is 'init' rather than '__init__'.
    @forward(*inputs): just like pytorch needed.
    @get_loss(*inputs, targets) : to support fetching the final loss from multi-loss.
    @get_posterior(): to compute accuracy.
    @extract_embedding(inputs) : needed if use pipline/onestep/extract_embeddings.py.
    """

    def __init__(self, *args, **kwargs):
        super(SpeakerNet, self).__init__()
        params_dict = locals()
        model_name = str(params_dict["self"]).split("()")[0]
        args_str = utils.iterator_to_params_str(params_dict["args"])
        kwargs_str = utils.dict_to_params_str(params_dict["kwargs"])

        self.model_creation = "{0}({1},{2})".format(model_name, args_str, kwargs_str)

        self.loss = None
        self.use_step = False
        self.transform_keys: List[str] = []
        self.rename_transform_keys: Dict[str, str] = {}
        self.init(*args, **kwargs)
        self._init_features()

    def _init_features(self):
        # Input features
        if self.features == "fbank":
            from speakernet.features.features import Fbank

            self.compute_features = Fbank(n_mels=self.inputs_dim)
        else:
            raise NotImplementedError
        # self.instancenorm = torch.nn.InstanceNorm1d(self.inputs_dim)

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def get_model_creation(self):
        return self.model_creation

    @utils.for_device_free
    def get_feats(self, wavs):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                feats = self.compute_features(wavs)
                feats = feats.transpose(1, 2)
                # CMN
                # feats = self.instancenorm(feats)
                feats = feats - torch.mean(feats, dim=2, keepdim=True)
                if self.norm_var:
                    var = torch.var(feats, dim=2, keepdim=True, unbiased=False)
                    feats = feats / torch.sqrt(var + 1e-8)

        return feats

    # You could use this decorator if needed in class function overwriting
    @utils.for_device_free
    def forward(self, *inputs):
        raise NotImplementedError

    # You could use this decorator if needed in class function overwriting
    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """
        @return: return a loss tensor, such as return from torch.nn.CrossEntropyLoss(reduction='mean')

        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        return self.loss(inputs, targets)

    def get_posterior(self):
        """
        Should call get_posterior after calling get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @utils.for_device_free
    def get_accuracy(self, targets):
        """
        @return: return accuracy
        """
        return self.loss.get_accuracy(targets)

    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not
        """
        return layer(x) if layer is not None else x

    def load_transform_state_dict(self, state_dict):
        """It is used in transform-learning.
        """
        assert type(self.transform_keys) == list
        assert type(self.rename_transform_keys) == dict
        if self.transform_keys != []:
            # # For large-margin finetuning or other situation
            # if (
            #     "loss" in self.transform_keys
            #     and state_dict["loss.weight"].size(0) == 3 * self.loss.num_targets
            # ):
            #     print("hhhhhhhhhhhhh")

            #     new_num_targets = self.loss.num_targets
            #     state_dict["loss.weight"] = state_dict["loss.weight"][:new_num_targets, :, :]
            remaining = {
                utils.key_to_value(self.rename_transform_keys, k, False): v
                for k, v in state_dict.items()
                if k.split(".")[0] in self.transform_keys or k in self.transform_keys
            }
            self.load_state_dict(remaining, strict=False)
        else:
            self.load_state_dict(state_dict)

        return self

    # We could use this decorator if needed when overwriting class function.
    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs):
        """ If use the decorator, should note:
        @inputs: a 3-dimensional tensor with batch-dim=1 or [frames, feature-dim] matrix for
                acoustic features only
        @return: an 1-dimensional vector
        """
        raise NotImplementedError

    def get_warmR_T(self, T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end) / (T_i - 1) * (T_cur % T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin.
        if self.use_step:
            if self.step_params["m"]:
                # The lambda is fixed as the initial value before increase_start_epoch
                current_position = epoch * epoch_batchs + this_iter + 1
                increase_start_epoch = self.step_params["increase_start_epoch"].split(".")
                if len(increase_start_epoch) == 2:
                    increase_start_position = int(increase_start_epoch[0]) * epoch_batchs \
                        + int(increase_start_epoch[1])
                elif len(increase_start_epoch) == 1:
                    increase_start_position = int(increase_start_epoch[0]) * epoch_batchs
                else:
                    raise ValueError

                if current_position >= increase_start_position:
                    current_position = current_position - increase_start_position
                    lambda_factor = max(
                        self.step_params["lambda_0"],
                        self.step_params["lambda_b"]
                        * (1 + self.step_params["gamma"] * current_position)
                        ** (-self.step_params["alpha"]),
                    )
                else:
                    lambda_factor = self.step_params["lambda_b"]
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (
                self.step_params["t"] or self.step_params["p"]
            ):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur * epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def backward_step(self, epoch, this_iter, epoch_batchs):
        pass


class SpeakerLoss(torch.nn.Module):
    """ This is a virtual loss class to be suitable for pipline scripts, such as train.py. And it requires
    to implement the function get_posterior to compute accuracy. But just using self.posterior to record the outputs
    before computing loss in forward is more convenient.
    For example,
        def forward(self, inputs, targets):
            outputs = softmax(inputs)
            self.posterior = outputs
            loss = CrossEntropy(outputs, targets)
        return loss
    It means that get_posterior should be called after forward.
    """

    def __init__(self, *args, **kwargs):
        super(SpeakerLoss, self).__init__()
        self.posterior = None
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError

    def get_posterior(self):
        assert self.posterior is not None
        return self.posterior

    @utils.for_device_free
    def get_accuracy(self, targets):
        """
        @return: return accuracy
        """
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(self.get_posterior(), dim=1))
            num_correct = (targets == prediction).sum()

        return num_correct.item() / len(targets)

    def step(self, lambda_factor):
        pass
