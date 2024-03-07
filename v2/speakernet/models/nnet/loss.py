# -*- coding:utf-8 -*-
"""
Copyright 2019 Snowdar
          2022 Jianchen Li
"""

import os
import sys
import math
import torch
import numpy as np
import torch.nn.functional as F

# sys.path.insert(0, "./")
from speakernet.models.nnet.components import SoftmaxAffineLayer
from speakernet.models.nnet.framework import SpeakerLoss
from speakernet.utils.utils import to_device, for_device_free


#############################################


# Loss ✿
"""
Note, there are some principles about loss implements:
    In process: torch.nn.CrossEntropyLoss = softmax + log + torch.nn.NLLLoss()
    In function: torch.nn.NLLLoss() <-> - (sum(torch.tensor.gather())
so, in order to keep codes simple and efficient, do not using 'for' or any other complex grammar to implement what could be replaced by above.
"""


class SoftmaxLoss(SpeakerLoss):
    """ An usual log-softmax loss with affine component.
    """

    def init(
        self,
        input_dim,
        num_targets,
        t=1,
        reduction="mean",
        special_init=False,
        label_smoothing=0.0,
    ):
        self.affine = torch.nn.Linear(input_dim, num_targets)
        torch.nn.init.normal_(self.affine.weight, 0.0, 0.01)  # It seems better.
        torch.nn.init.constant_(self.affine.bias, 0.0)
        self.t = t  # temperature
        # CrossEntropyLoss() has included the LogSoftmax, so do not add this function extra.
        print(label_smoothing)
        self.loss_function = torch.nn.CrossEntropyLoss(
            reduction=reduction, label_smoothing=label_smoothing
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        # The special_init is not recommended in this loss component
        if special_init:
            torch.nn.init.xavier_normal_(self.affine.weight, gain=1.0)
            # torch.nn.init.xavier_uniform_(self.affine.weight, gain=1.0)

    @for_device_free
    def forward(self, inputs, targets):
        outputs = self.affine(inputs.squeeze(2))
        self.posterior = outputs.detach()

        return self.loss_function(outputs / self.t, targets)


class SoftmaxLossAffine(SpeakerLoss):
    """ An usual log-softmax loss with affine component.
    """

    def init(self, input_dim, num_targets, special_init=False):
        self.affine = torch.nn.Linear(input_dim, num_targets)
        torch.nn.init.normal_(self.affine.weight, 0.0, 0.01)  # It seems better.
        torch.nn.init.constant_(self.affine.bias, 0.0)

        # The special_init is not recommended in this loss component
        if special_init:
            torch.nn.init.xavier_normal_(self.affine.weight, gain=1.0)
            # torch.nn.init.xavier_uniform_(self.affine.weight, gain=1.0)

    @for_device_free
    def forward(self, inputs):
        outputs = self.affine(inputs.squeeze(2))
        return outputs


class SoftmaxLossFunc(SpeakerLoss):
    """ An usual log-softmax loss with criterion.
    """

    def init(
        self,
        t=1,
        reduction="mean",
        label_smoothing=0.0,
        soft_target=False
    ):
        self.t = t  # temperature
        self.soft_target = soft_target
        # CrossEntropyLoss() has included the LogSoftmax, so do not add this function extra.
        self.loss_function = torch.nn.CrossEntropyLoss(
            reduction=reduction, label_smoothing=label_smoothing
        )
        print(label_smoothing)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    @for_device_free
    def forward(self, logits, targets):
        self.posterior = logits.detach()

        if self.soft_target:
            log_probs = self.logsoftmax(logits)
            loss = (-F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
            return loss

        return self.loss_function(logits / self.t, targets)


class FocalLoss(SpeakerLoss):
    """Implement focal loss according to [Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.
    "Focal loss for dense object detection", IEEE international conference on computer vision, 2017.]
    """

    def init(self, input_dim, num_targets, gamma=2, reduction="sum", eps=1.0e-10):

        self.softmax_affine = SoftmaxAffineLayer(
            input_dim, num_targets, dim=1, log=False, bias=True
        )
        self.loss_function = torch.nn.NLLLoss(reduction=reduction)

        self.gamma = gamma
        # self.alpha = alpha
        self.eps = eps

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        posterior = self.softmax_affine(inputs)
        self.posterior = posterior.detach()

        focal_posterior = (1 - posterior) ** self.gamma * torch.log(posterior.clamp(min=self.eps))
        outputs = torch.squeeze(focal_posterior, dim=2)
        return self.loss_function(outputs, targets)


class MarginSoftmaxLoss(SpeakerLoss):
    """Margin softmax loss.
    Support AM, AAM, Double-AM.
    Do not provide A-softmax loss again for its complex implementation and margin limitation.
    Reference:
            [1] Liu, W., Wen, Y., Yu, Z., & Yang, M. (2016). Large-margin softmax loss for convolutional neural networks.
                Paper presented at the ICML.

            [2] Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., & Song, L. (2017). Sphereface: Deep hypersphere embedding for
                face recognition. Paper presented at the Proceedings of the IEEE conference on computer vision and pattern
                recognition.

            [3] Wang, F., Xiang, X., Cheng, J., & Yuille, A. L. (2017). Normface: l2 hypersphere embedding for face
                verification. Paper presented at the Proceedings of the 25th ACM international conference on Multimedia.

            [4] Wang, F., Cheng, J., Liu, W., & Liu, H. (2018). Additive margin softmax for face verification. IEEE Signal
                Processing Letters, 25(7), 926-930.

            [5] Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., . . . Liu, W. (2018). Cosface: Large margin cosine
                loss for deep face recognition. Paper presented at the Proceedings of the IEEE Conference on Computer Vision
                and Pattern Recognition.

            [6] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face
                recognition. Paper presented at the Proceedings of the IEEE Conference on Computer Vision and Pattern
                Recognition.

            [7] Zhou, S., Chen, C., Han, G., & Hou, X. (2020). Double Additive Margin Softmax Loss for Face Recognition.
                Applied Sciences, 10(1), 60.
    """

    def init(
        self,
        input_dim,
        num_targets,
        m=0.2,
        s=30.0,
        t=1.0,
        K=1,
        feature_normalize=True,
        method="am",
        double=False,
        mhe_loss=False,
        mhe_w=0.01,
        inter_loss=0.0,
        ring_loss=0.0,
        curricular=False,
        reduction="mean",
        eps=1.0e-10,
        init=True,
        easy_margin=False,
    ):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.s = s  # scale factor with feature normalization
        self.m = m  # margin
        self.t = t  # temperature
        self.K = K  # subcenter
        self.weight = torch.nn.Parameter(torch.randn(num_targets * self.K, input_dim, 1))
        self.feature_normalize = feature_normalize
        self.method = method  # am | aam
        self.double = double
        self.feature_normalize = feature_normalize
        self.mhe_loss = mhe_loss
        self.mhe_w = mhe_w
        self.inter_loss = inter_loss
        self.ring_loss = ring_loss
        self.lambda_factor = 0

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.mmm = 1.0 + math.cos(math.pi - self.m)  # this can make the output more continuous

        self.curricular = CurricularMarginComponent() if curricular else None

        if self.ring_loss > 0:
            self.r = torch.nn.Parameter(torch.tensor(20.0))
            self.feature_normalize = False

        self.eps = eps

        if feature_normalize:
            p_target = [0.9, 0.95, 0.99]
            suggested_s = [
                (num_targets - 1) / num_targets * np.log((num_targets - 1) * x / (1 - x))
                for x in p_target
            ]

            if self.s < suggested_s[0]:
                print(
                    "Warning : using feature noamlization with small scalar s={s} could result in bad convergence. \
                There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
                        s=self.s, suggested_s=suggested_s, p_target=p_target
                    )
                )

        # label_smoothing does not work for MarginSoftmaxLoss
        # See: https://zhuanlan.zhihu.com/p/302843504
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        # Init weight.
        if init:
            # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0.0, 0.01)  # It seems better.

    def forward(self, inputs, targets):
        """
        @inputs: a 2-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        # Normalize
        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine = F.linear(normalized_x, normalized_weight)  # Y = W*X

        # Subcenter
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.num_targets, self.K))
            cosine, _ = torch.max(cosine, dim=2)

        if not self.feature_normalize:
            self.s = inputs.norm(2, dim=1)  # [batch-size, l2-norm]
            # The accuracy must be reported before margin penalty added
            self.posterior = (self.s.detach() * cosine.detach()).unsqueeze(2)
        else:
            self.posterior = (self.s * cosine.detach()).unsqueeze(2)

        if not self.training:
            # For valid set.
            outputs = self.s * cosine
            return self.loss_function(outputs, targets)

        # Margin Penalty
        # cosine [batch_size, num_class]
        # targets.unsqueeze(1) [batch_size, 1]
        cosine_target = cosine.gather(1, targets.unsqueeze(1))

        if self.inter_loss > 0:
            inter_cosine = torch.softmax(self.s * cosine, dim=1)
            inter_cosine_target = inter_cosine.gather(1, targets.unsqueeze(1))
            # 1/N * \log (\frac{1 - logit_y}{C-1})
            inter_loss = torch.log(
                (inter_cosine.sum(dim=1) - inter_cosine_target) / (self.num_targets - 1) + self.eps
            ).mean()

        if self.method == "am":
            phi = cosine_target - self.m
            if self.double:
                double_cosine = cosine + self.m
        elif self.method == "aam":
            cosine = cosine.float()
            cosine_target = cosine_target.float()
            # phi = torch.cos(torch.acos(cosine_target) + self.m)

            # Another implementation w.r.t cosine(theta+m) = cosine * cos_m - sin_theta * sin_m
            phi = self.cos_m * cosine_target - self.sin_m * torch.sqrt(
                (1 - torch.pow(cosine_target, 2)).clamp(0, 1)
            )
            if self.easy_margin:
                phi = torch.where(cosine_target > 0, phi, cosine_target)
            else:
                # phi = torch.where(cosine_target > self.th, phi, cosine_target - self.mm)
                phi = torch.where(cosine_target > self.th, phi, cosine_target - self.mmm,)

            if self.double:
                double_cosine = torch.cos(torch.acos(cosine).add(-self.m))
        else:
            raise ValueError(
                "Do not support this {0} margin w.r.t [ am | aam ]".format(self.method)
            )

        # 模拟退火
        phi = (
            1 / (1 + self.lambda_factor) * phi
            + self.lambda_factor / (1 + self.lambda_factor) * cosine_target
        )

        if self.double:
            cosine = (
                1 / (1 + self.lambda_factor) * double_cosine
                + self.lambda_factor / (1 + self.lambda_factor) * cosine
            )

        if self.curricular is not None:
            cosine = self.curricular(cosine, cosine_target, phi)

        outputs = self.s * cosine.scatter(1, targets.unsqueeze(1), phi)

        # Other extra loss
        # Final reported loss will be always higher than softmax loss for the absolute margin penalty and
        # it is a lie about why we can not decrease the loss to a mininum value. We should not report the
        # loss after margin penalty did but we really report this invalid loss to avoid computing the
        # training loss twice.

        if self.ring_loss > 0:
            ring_loss = torch.mean((self.s - self.r) ** 2) / 2
        else:
            ring_loss = 0.0

        if self.mhe_loss:
            sub_weight = normalized_weight - torch.index_select(
                normalized_weight, 0, targets
            ).unsqueeze(dim=1)
            # [N, C]
            normed_sub_weight = sub_weight.norm(2, dim=2)
            mask = torch.full_like(normed_sub_weight, True, dtype=torch.bool).scatter_(
                1, targets.unsqueeze(dim=1), False
            )
            # [N, C-1]
            normed_sub_weight_clean = torch.masked_select(normed_sub_weight, mask).reshape(
                targets.size()[0], -1
            )
            # torch.mean means 1/(N*(C-1))
            the_mhe_loss = self.mhe_w * torch.mean(
                (normed_sub_weight_clean ** 2).clamp(min=self.eps) ** -1
            )

            return (
                self.loss_function(outputs / self.t, targets)
                + the_mhe_loss
                + self.ring_loss * ring_loss
            )
        elif self.inter_loss > 0:
            return (
                self.loss_function(outputs / self.t, targets)
                + self.inter_loss * inter_loss
                + self.ring_loss * ring_loss
            )
        else:
            return self.loss_function(outputs / self.t, targets) + self.ring_loss * ring_loss

    def step(self, lambda_factor):
        self.lambda_factor = lambda_factor

    def extra_repr(self):
        return (
            "(~affine): (input_dim={input_dim}, num_targets={num_targets}, method={method}, double={double}, "
            "margin={m}, s={s}, t={t}, feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, "
            "eps={eps})".format(**self.__dict__)
        )


class MarginSoftmaxAffine(SpeakerLoss):
    """Compute the logit as input to the loss function. **Split the computations of logit and loss function**
    Support AM, AAM.
    Do not provide A-softmax loss again for its complex implementation and margin limitation.
    """

    def init(
        self, input_dim, num_targets, K=1, init=True,
    ):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.K = K  # subcenter
        self.weight = torch.nn.Parameter(torch.randn(num_targets * self.K, input_dim, 1))

        # Init weight.
        if init:
            # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0.0, 0.01)  # It seems better.

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        # Normalize
        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine = F.linear(normalized_x, normalized_weight)  # Y = W*X

        # Subcenter
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.num_targets, self.K))
            cosine, _ = torch.max(cosine, dim=2)

        return cosine


class MarginSoftmaxLossFunc(SpeakerLoss):
    """Margin softmax loss criterion. **Split the computations of logit and loss function**
    Support AM, AAM.
    Do not provide A-softmax loss again for its complex implementation and margin limitation.
    """

    def init(
        self,
        m=0.2,
        s=30.0,
        t=1.0,
        method="am",
        reduction="mean",
        eps=1.0e-10,
        easy_margin=False,
        soft_target=False,
    ):

        self.m = m  # margin
        self.t = t  # temperature
        self.s = s  # scale factor with feature normalization
        self.method = method  # am | aam
        self.lambda_factor = 0
        self.soft_target = soft_target

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.mmm = 1.0 + math.cos(math.pi - self.m)  # this can make the output more continuous

        self.eps = eps

        # label_smoothing does not work for MarginSoftmaxLoss
        # See: https://zhuanlan.zhihu.com/p/302843504
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.0)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, cosine, targets):
        self.posterior = (self.s * cosine.detach()).unsqueeze(2)

        if not self.training:
            # For valid set.
            outputs = self.s * cosine
            return self.loss_function(outputs, targets)

        if self.soft_target:
            soft_targets = targets
            _, targets = torch.max(targets, dim=1)

        # Margin Penalty
        # cosine [batch_size, num_class]
        # targets.unsqueeze(1) [batch_size, 1]
        cosine_target = cosine.gather(1, targets.unsqueeze(1))

        if self.method == "am":
            phi = cosine_target - self.m
        elif self.method == "aam":
            cosine = cosine.float()
            cosine_target = cosine_target.float()
            # phi = torch.cos(torch.acos(cosine_target) + self.m)

            # Another implementation w.r.t cosine(theta+m) = cosine * cos_m - sin_theta * sin_m
            phi = self.cos_m * cosine_target - self.sin_m * torch.sqrt(
                (1 - torch.pow(cosine_target, 2)).clamp(0, 1)
            )
            if self.easy_margin:
                phi = torch.where(cosine_target > 0, phi, cosine_target)
            else:
                # phi = torch.where(cosine_target > self.th, phi, cosine_target - self.mm)
                phi = torch.where(cosine_target > self.th, phi, cosine_target - self.mmm,)
        else:
            raise ValueError(
                "Do not support this {0} margin w.r.t [ am | aam ]".format(self.method)
            )

        # 模拟退火
        phi = (
            1 / (1 + self.lambda_factor) * phi
            + self.lambda_factor / (1 + self.lambda_factor) * cosine_target
        )

        outputs = self.s * cosine.scatter(1, targets.unsqueeze(1), phi)

        # if self.soft_target:
        #     log_probs = self.logsoftmax(outputs / self.t)
        #     loss = (-F.softmax(soft_targets, dim=1).detach() * log_probs).mean(0).sum()
        #     return loss, targets
        # else:
        return self.loss_function(outputs / self.t, targets)

    def step(self, lambda_factor):
        self.lambda_factor = lambda_factor

    def get_accuracy(self, targets):
        """
        @return: return accuracy
        """
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(self.get_posterior(), dim=1))
            num_correct = (targets == prediction).sum()

        return num_correct.item() / len(targets)


class CurricularMarginComponent(torch.nn.Module):
    """CurricularFace is implemented as a called component for MarginSoftmaxLoss.
    Reference: Huang, Yuge, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li,
               and Feiyue Huang. 2020. “CurricularFace: Adaptive Curriculum Learning Loss for Deep Face
               Recognition.” ArXiv E-Prints arXiv:2004.00288.
    Github: https://github.com/HuangYG123/CurricularFace. Note, the momentum of this github is a wrong value w.r.t
            the above paper. The momentum 't' should not increase so fast and I have corrected it as follow.

    By the way, it does not work in my experiments.
    """

    def __init__(self, momentum=0.01):
        super(CurricularMarginComponent, self).__init__()
        self.momentum = momentum
        self.register_buffer("t", torch.zeros(1))

    def forward(self, cosine, cosine_target, phi):
        with torch.no_grad():
            self.t = (1 - self.momentum) * cosine_target.mean() + self.momentum * self.t

        mask = cosine > phi
        hard_example = cosine[mask]
        # Use clone to avoid problem "RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation"
        cosine_clone = cosine.clone()
        cosine_clone[mask] = hard_example * (self.t + hard_example)

        return cosine_clone


class LogisticAffinityLoss(SpeakerLoss):
    """LogisticAffinityLoss.
    Reference: Peng, J., Gu, R., & Zou, Y. (2019).
               LOGISTIC SIMILARITY METRIC LEARNING VIA AFFINITY MATRIX FOR TEXT-INDEPENDENT SPEAKER VERIFICATION.
    """

    def init(self, init_w=5.0, init_b=-1.0, reduction="mean"):
        self.reduction = reduction

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))

    def forward(self, inputs, targets):
        # This loss has no way to compute accuracy
        S = F.normalize(inputs.squeeze(dim=2), dim=1)
        # This can not keep the diag-value equal to 1 and it maybe a question.
        A = torch.sigmoid(self.w * torch.mm(S, S.t()) + self.b)

        targets_matrix = targets + torch.zeros_like(A)
        condition = targets_matrix - targets_matrix.t()
        outputs = -torch.log(torch.where(condition == 0, A, 1 - A))

        if self.reduction == "sum":
            return outputs.sum()
        elif self.reduction == "mean":
            return outputs.sum() / targets.shape[0]
        else:
            raise ValueError("Do not support this reduction {0}".format(self.reduction))


class MixupLoss(SpeakerLoss):
    """Implement a mixup component to augment data and increase the generalization of model training.
    Reference:
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. n.d. Mixup: BEYOND EMPIRICAL RISK MINIMIZATION.
        [2] Zhu, Yingke, Tom Ko, and Brian Mak. 2019. “Mixup Learning Strategies for Text-Independent Speaker Verification.”

    Github: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    """

    def init(self, base_loss, mixup_component):

        self.base_loss = base_loss
        self.mixup_component = mixup_component

    def forward(self, inputs, targets):
        if self.training:
            lam = self.mixup_component.lam
            index = self.mixup_component.index

            loss = lam * self.base_loss(inputs, targets) + (1 - lam) * self.base_loss(
                inputs, targets[index]
            )
        else:
            loss = self.base_loss(inputs, targets)

        return loss

    def get_accuracy(self, targets):
        if self.training:
            # It is not very clear to compute accuracy for mixed data.
            lam = self.mixup_component.lam
            index = self.mixup_component.index
            return lam * self.compute_accuracy(self.base_loss.get_posterior(), targets) + (
                1 - lam
            ) * self.compute_accuracy(self.base_loss.get_posterior(), targets[index])
        else:
            return self.compute_accuracy(self.base_loss.get_posterior(), targets)


class AngularProto(SpeakerLoss):
    def init(self, num_per_spk=2, init_w=10.0, init_b=-5.0, label_smoothing=0.0):
        self.num_per_spk = num_per_spk

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))
        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, targets=None):
        inputs = inputs.squeeze(-1)
        assert len(inputs.size()) == 2
        inputs = inputs.reshape(-1, self.num_per_spk, inputs.size(-1))
        assert inputs.size(1) >= 2

        out_anchor = torch.mean(inputs[:, 1:, :], 1)
        out_positive = inputs[:, 0, :]
        stepsize = out_anchor.size(0)

        # TODO
        cos_sim_matrix = F.cosine_similarity(
            out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2)
        )
        torch.clamp(self.w, 1e-6)
        self.posterior = cos_sim_matrix.detach()
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()
        return self.loss_function(cos_sim_matrix, label)

    @for_device_free
    def get_accuracy(self, targets=None):
        targets = torch.from_numpy(np.asarray(range(0, self.get_posterior().size(0)))).cuda()
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(self.get_posterior(), dim=1))
            num_correct = (targets == prediction).sum()

        return num_correct.item() / len(targets)


class Prototypical(SpeakerLoss):
    def init(self, num_per_spk=2, label_smoothing=0.0):
        self.num_per_spk = num_per_spk

        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, targets=None):
        inputs = inputs.squeeze(-1)
        assert len(inputs.size()) == 2
        inputs = inputs.reshape(-1, self.num_per_spk, inputs.size(-1))
        assert inputs.size(1) >= 2

        out_anchor = torch.mean(inputs[:, 1:, :], dim=1)
        out_positive = inputs[:, 0, :]
        num_spks = out_anchor.size(0)

        # p_norm = torch.pow(out_positive, 2).sum(1, keepdim=True).expand(num_spks, num_spks)
        # a_norm = torch.pow(out_anchor, 2).sum(1, keepdim=True).expand(num_spks, num_spks).t()
        # output = p_norm + a_norm
        # output.addmm_(1, -2, out_positive, out_anchor.t())
        # output = -1 * output
        output = torch.mm(out_positive, out_anchor.t())

        self.posterior = output.detach()

        label = torch.from_numpy(np.asarray(range(0, num_spks))).cuda()
        return self.loss_function(output, label)

    @for_device_free
    def get_accuracy(self, targets=None):
        targets = torch.from_numpy(np.asarray(range(0, self.get_posterior().size(0)))).cuda()
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(self.get_posterior(), dim=1))
            num_correct = (targets == prediction).sum()

        return num_correct.item() / len(targets)


if __name__ == "__main__":
    torch.manual_seed(1024)
    # loss = MarginSoftmaxLoss(256, 5994, K=3)
    loss = SoftmaxLoss(256, 5994)
    torch.manual_seed(1024)
    loss_affine = SoftmaxLossAffine(256, 5994)
    loss_func = SoftmaxLossFunc()
    a = torch.rand((2, 256, 1))
    target = torch.LongTensor([0, 1])
    out = loss(a, target)

    out1 = loss_func(loss_affine(a), target)
    print(out)
    print(out1)
