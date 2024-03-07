"""
Copyright 2019 Snowdar
          2022 Jianchen Li
"""

import numpy as np

import torch
import torch.nn.functional as F

from .activation import Nonlinearity

from speakernet.utils.utils import to_device, assign_params_dict


### There are some basic custom components/layers. ###

# Base ✿
class FTdnnBlock(torch.nn.Module):
    """ Factorized TDNN block w.r.t http://danielpovey.com/files/2018_interspeech_tdnnf.pdf.
    Reference: Povey, D., Cheng, G., Wang, Y., Li, K., Xu, H., Yarmohammadi, M., & Khudanpur, S. (2018). 
               Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks. Paper presented at the Interspeech.
    """
    def __init__(self, input_dim, output_dim, bottleneck_dim, context_size=0, bypass_scale=0.66, pad=True):
        super(FTdnnBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.context_size = context_size
        self.bypass_scale = bypass_scale
        self.pad = pad

        if context_size > 0:
            context_factor1 = [-context_size, 0]
            context_factor2 = [0, context_size]
        else:
            context_factor1 = [0]
            context_factor2 = [0]

        self.factor = TdnnAffine(input_dim, bottleneck_dim, context_factor1, pad=pad, bias=False)
        self.affine = TdnnAffine(bottleneck_dim, output_dim, context_factor2, pad=pad, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(output_dim, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        identity = inputs

        out = self.factor(inputs)
        out = self.affine(out)
        out = self.relu(out)
        out = self.bn(out)
        if self.bypass_scale != 0:
            # assert identity.shape[1] == self.output_dim
            out += self.bypass_scale * identity

        return out


    '''
    Reference: https://github.com/cvqluu/Factorized-TDNN.
    '''
    def step_semi_orth(self):
        with torch.no_grad():
            M = self.get_semi_orth_weight(self.factor)
            self.factor.weight.copy_(M)

    @staticmethod
    def get_semi_orth_weight(conv1dlayer):
        # updates conv1 weight M using update rule to make it more semi orthogonal
        # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
        # includes the tweaks related to slowing the update speed
        # only an implementation of the 'floating scale' case
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = conv1dlayer.weight.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            assert ratio > 0.99
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5

            scale2 = trace_PP/trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            updated = M + update
            # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return updated.reshape(*orig_shape) if mshape[0] > mshape[1] else updated.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1]*orig_shape[2], orig_shape[0])


# Block ✿
class SoftmaxAffineLayer(torch.nn.Module):
    """ An usual 2-fold softmax layer with an affine transform.
    @dim: which dim to apply softmax on
    """

    def __init__(self, input_dim, output_dim, context=[0], dim=1, log=True, bias=True, groups=1, t=1., special_init=False):
        super(SoftmaxAffineLayer, self).__init__()

        self.affine = TdnnAffine(
            input_dim, output_dim, context=context, bias=bias, groups=groups)
        # A temperature parameter.
        self.t = t

        if log:
            self.softmax = torch.nn.LogSoftmax(dim=dim)
        else:
            self.softmax = torch.nn.Softmax(dim=dim)

        if special_init:
            torch.nn.init.xavier_uniform_(
                self.affine.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs):
        """
        @inputs: any, such as a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        return self.softmax(self.affine(inputs)/self.t)


# ReluBatchNormLayer
class _BaseActivationBatchNorm(torch.nn.Module):
    """[Affine +] Relu + BatchNorm1d.
    Affine could be inserted by a child class.
    """

    def __init__(self):
        super(_BaseActivationBatchNorm, self).__init__()
        self.affine = None
        self.activation = None
        self.batchnorm = None

    def add_relu_bn(self, output_dim=None, options: dict = {}):
        default_params = {
            "bn-relu": False,
            "nonlinearity": 'relu',
            "nonlinearity_params": {"inplace": True, "negative_slope": 0.01},
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True},
        }

        default_params = assign_params_dict(default_params, options)
        self.bn_relu = default_params["bn-relu"]

        # This 'if else' is used to keep a corrected order when printing model.
        # torch.sequential is not used for I do not want too many layer wrappers and just keep structure as tdnn1.affine
        # rather than tdnn1.layers.affine or tdnn1.layers[0] etc..
        if not default_params["bn-relu"]:
            # ReLU-BN
            # For speaker recognition, relu-bn seems better than bn-relu. And w/o affine (scale and shift) of bn is
            # also better than w/ affine.
            # Assume the activation function has no parameters
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
        else:
            # BN-ReLU
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])

    def _bn_relu_forward(self, x):
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            # Assume the activation function has no parameters
            x = self.activation(x)
        return x

    def _relu_bn_forward(self, x):
        if self.activation is not None:
            # Assume the activation function has no parameters
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = self.affine(inputs)
        if not self.bn_relu:
            outputs = self._relu_bn_forward(x)
        else:
            outputs = self._bn_relu_forward(x)
        return outputs


class Conv1dBnReluLayer(_BaseActivationBatchNorm):
    """ Conv1d-BN-Relu
    """

    def __init__(self, input_dim, output_dim, context=[0], **options):
        super(Conv1dBnReluLayer, self).__init__()

        affine_options = {
            "bias": True,
            "groups": 1,
        }

        affine_options = assign_params_dict(affine_options, options)

        # Only keep the order: affine -> layers.insert -> add_relu_bn,
        # the structure order will be:
        # (fc2): Conv1dBnReluLayer(
        #        (affine): Conv1d()
        #        (activation): ReLU()
        #        (batchnorm): BatchNorm1d()
        dilation = context[1] - context[0] if len(context) > 1 else 1
        for i in range(1, len(context) - 1):
            assert dilation == context[i + 1] - context[i]

        left_context = context[0] if context[0] < 0 else 0
        right_context = context[-1] if context[-1] > 0 else 0
        receptive_field_size = right_context - left_context + 1
        padding = receptive_field_size // 2

        self.affine = torch.nn.Conv1d(input_dim, output_dim, kernel_size=len(context),
                                      stride=1, padding=padding, dilation=dilation,
                                      groups=affine_options["groups"],
                                      bias=affine_options["bias"])

        self.add_relu_bn(output_dim, options=options)

        # Implement forward function extrally if needed when forward-graph is changed.


class ReluBatchNormTdnnfLayer(_BaseActivationBatchNorm):
    """ F-TDNN-ReLU-BN.
    An usual 3-fold layer with TdnnfBlock affine.
    """

    def __init__(self, input_dim, output_dim, inner_size, context_size=0, **options):
        super(ReluBatchNormTdnnfLayer, self).__init__()

        self.affine = TdnnfBlock(
            input_dim, output_dim, inner_size, context_size)
        self.add_relu_bn(output_dim, options=options)


class AdaptivePCMN(torch.nn.Module):
    """ Using adaptive parametric Cepstral Mean Normalization to replace traditional CMN.
        It is implemented according to [Ozlem Kalinli, etc. "Parametric Cepstral Mean Normalization 
        for Robust Automatic Speech Recognition", icassp, 2019.]
    """

    def __init__(self, input_dim, left_context=-10, right_context=10, pad=True):
        super(AdaptivePCMN, self).__init__()

        assert left_context < 0 and right_context > 0

        self.left_context = left_context
        self.right_context = right_context
        self.tot_context = self.right_context - self.left_context + 1

        kernel_size = (self.tot_context,)

        self.input_dim = input_dim
        # Just pad head and end rather than zeros using replicate pad mode
        # or set pad false with enough context egs.
        self.pad = pad
        self.pad_mode = "replicate"

        self.groups = input_dim
        output_dim = input_dim

        # The output_dim is equal to input_dim and keep every dims independent by using groups conv.
        self.beta_w = torch.nn.Parameter(torch.randn(
            output_dim, input_dim//self.groups, *kernel_size))
        self.alpha_w = torch.nn.Parameter(torch.randn(
            output_dim, input_dim//self.groups, *kernel_size))
        self.mu_n_0_w = torch.nn.Parameter(torch.randn(
            output_dim, input_dim//self.groups, *kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))

        # init weight and bias. It is important
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.beta_w, 0., 0.01)
        torch.nn.init.normal_(self.alpha_w, 0., 0.01)
        torch.nn.init.normal_(self.mu_n_0_w, 0., 0.01)
        torch.nn.init.constant_(self.bias, 0.)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        assert inputs.shape[2] >= self.tot_context

        if self.pad:
            pad_input = F.pad(inputs, (-self.left_context,
                                       self.right_context), mode=self.pad_mode)
        else:
            pad_input = inputs
            inputs = inputs[:, :, -self.left_context:-self.right_context]

        # outputs beta + 1 instead of beta to avoid potentially zeroing out the inputs cepstral features.
        self.beta = F.conv1d(pad_input, self.beta_w,
                             bias=self.bias, groups=self.groups) + 1
        self.alpha = F.conv1d(pad_input, self.alpha_w,
                              bias=self.bias, groups=self.groups)
        self.mu_n_0 = F.conv1d(pad_input, self.mu_n_0_w,
                               bias=self.bias, groups=self.groups)

        outputs = self.beta * inputs - self.alpha * self.mu_n_0

        return outputs


class SEBlock1d(torch.nn.Module):
    """ A SE Block layer which can learn to use global information to selectively emphasise informative 
    features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
       Snowdar xmuspeech 2020-04-28 [Check and update]
       lijianchen 2020-11-18
    """

    def __init__(self, input_dim, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the SE blocks 
        in the network.
        '''
        super(SEBlock1d, self).__init__()

        self.input_dim = input_dim

        self.fc_1 = torch.nn.Linear(input_dim, input_dim // ratio, bias=False)
        self.fc_2 = torch.nn.Linear(input_dim // ratio, input_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.fc_1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.xavier_normal_(self.fc_2.weight, gain=1.0)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        x = inputs.mean(dim=2, keepdim=True)
        x = self.relu(self.fc_1(x))
        scale = self.sigmoid(self.fc_2(x))

        return inputs * scale


class Mixup(torch.nn.Module):
    """Implement a mixup component to augment data and increase the generalization of model training.
    Reference:
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. n.d. Mixup: BEYOND EMPIRICAL RISK MINIMIZATION.
        [2] Zhu, Yingke, Tom Ko, and Brian Mak. 2019. “Mixup Learning Strategies for Text-Independent Speaker Verification.”

    Github: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    """

    def __init__(self, alpha=1.0):
        super(Mixup, self).__init__()

        self.alpha = alpha

    def forward(self, inputs):
        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        self.lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        # Shuffle the original index to generate the pairs, such as
        # Origin:           1 2 3 4 5
        # After Shuffling:  3 4 1 5 2
        # Then the pairs are (1, 3), (2, 4), (3, 1), (4, 5), (5,2).
        self.index = torch.randperm(batch_size, device=inputs.device)

        mixed_data = self.lam * inputs + (1 - self.lam) * inputs[self.index, :]

        return mixed_data
