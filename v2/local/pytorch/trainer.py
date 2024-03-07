# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import sys
import time
import torch
import logging
import traceback
import progressbar
from rich import print

import speakernet.utils.utils as utils
from speakernet.training.trainer import Trainer
from speakernet.training.reporter import Reporter

# Wrap stderr before logger init.
progressbar.streams.wrap_stderr()

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)


class Trainer(Trainer):
    def compute_forward(self, s_batch, t_batch):
        return self.fit_CDMA(s_batch, t_batch)

    def fit_CDMA(self, s_batch, t_batch):
        s_inputs, _, s_targets, s_utts = s_batch
        t_inputs, _, t_targets, t_utts = t_batch

        inputs = torch.cat((s_inputs, t_inputs))
        batch_size = s_inputs.size(0)

        inputs = self.encoder.get_feats(inputs)
        feats = self.encoder_forward(inputs)
        s_feats = feats[:batch_size]
        t_feats = feats[batch_size:]

        assert s_targets.max() < 5994 * 3
        # assert t_targets.max() < 200 * 3
        with torch.cuda.amp.autocast(enabled=False):
            s_loss = self.encoder.get_loss(s_feats.float(), s_targets)
            s_acc = self.encoder.get_accuracy(s_targets) if self.compute_accuracy else None

        with torch.cuda.amp.autocast(enabled=False):
            (
                mmd_wc_loss,
                mmd_bc_loss,
                mmd_wc_hard_loss,
                mmd_bc_hard_loss,
                mmd_wsbt_loss,
                mmd_bswt_loss,
            ) = self.encoder.get_mmd_loss(s_feats.float(), t_feats.float())
        mmd_wc_loss = 2 * mmd_wc_loss
        mmd_bc_loss = 2 * mmd_bc_loss
        mmd_wc_hard_loss = 2 * mmd_wc_hard_loss
        mmd_bc_hard_loss = 2 * mmd_bc_hard_loss
        mmd_wsbt_loss = 0.2 * mmd_wsbt_loss
        mmd_bswt_loss = 0.2 * mmd_bswt_loss

        loss = s_loss + mmd_wc_loss + mmd_bc_loss + mmd_wc_hard_loss + mmd_bc_hard_loss - mmd_wsbt_loss - mmd_bswt_loss

        return (
            loss,
            {
                "s_loss": s_loss.item(),
                "s_acc": s_acc * 100,
                "mmd_wc_loss": mmd_wc_loss.item(),
                "mmd_bc_loss": mmd_bc_loss.item(),
                "mmd_wc_h_loss": mmd_wc_hard_loss.item(),
                "mmd_bc_h_loss": mmd_bc_hard_loss.item(),
            },
        )

    def fit(self):
        """Main function to start a training process.
        """
        try:
            self.init_training()

            if utils.is_main_training():
                self.reporter = Reporter(self)

            # For lookahead.
            if getattr(self.optimizer, "optimizer", None) is not None:
                self.optimizer = self.optimizer.optimizer
            self.last_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

            unfreeze_epoch = self.params["unfreeze_epoch"].split(".")
            if len(unfreeze_epoch) == 2:
                self.unfreeze_iter = int(unfreeze_epoch[0]) * self.data.num_batch_train + int(
                    unfreeze_epoch[1]
                )
            elif len(unfreeze_epoch) == 1:
                self.unfreeze_iter = int(unfreeze_epoch[0]) * self.data.num_batch_train
            else:
                raise ValueError

            for self.current_epoch in range(self.resume_epoch, self.epochs):
                if isinstance(
                    self.data.s_train_loader.sampler,
                    torch.utils.data.distributed.DistributedSampler,
                ) or hasattr(self.data.s_train_loader.sampler, "set_epoch"):
                    self.data.s_train_loader.sampler.set_epoch(self.current_epoch)

                if isinstance(
                    self.data.t_train_loader.sampler,
                    torch.utils.data.distributed.DistributedSampler,
                ) or hasattr(self.data.t_train_loader.sampler, "set_epoch"):
                    self.data.t_train_loader.sampler.set_epoch(self.current_epoch)

                self.data.s_generator.manual_seed(self.data.seed + self.current_epoch)
                self.data.t_generator.manual_seed(self.data.seed + self.current_epoch + 1)

                s_worker_state = {}
                t_worker_state = {}

                last_ckpt_time = time.time()

                for self.current_iter, (s_batch, t_batch) in enumerate(
                    zip(self.data.s_train_loader, self.data.t_train_loader), self.resume_iter,
                ):
                    if self.data.s_train_loader.num_workers > 0:
                        # The loader return the batch in the order of worker_id (cyclic)
                        _worker_state = s_batch[-1]

                        # A batch of data is from the same worker
                        s_worker_state[_worker_state["worker_id"]] = _worker_state
                        self.data.s_train_loader.dataset.worker_state = s_worker_state
                    s_batch = s_batch[:-1]

                    if self.data.t_train_loader.num_workers > 0:
                        # The loader return the batch in the order of worker_id (cyclic)
                        _worker_state = t_batch[-1]

                        # A batch of data is from the same worker
                        t_worker_state[_worker_state["worker_id"]] = _worker_state
                        self.data.t_train_loader.dataset.worker_state = t_worker_state
                    t_batch = t_batch[:-1]

                    self.training_point = (
                        self.current_epoch,
                        self.current_iter,
                        self.data.num_batch_train,
                    )  # It is important for reporter.

                    current_iter = (
                        self.training_point[0] * self.training_point[2] + self.training_point[1] + 1
                    )
                    if current_iter == self.unfreeze_iter:
                        print(current_iter)
                        # self.encoder.loss.lambda_factor = 0
                        # self.encoder.t_loss.lambda_factor = 0

                        for p in self.encoder.resnet.parameters():
                            p.requires_grad = True
                        for p in self.encoder.stats.parameters():
                            p.requires_grad = True
                        for p in self.encoder.fc2.parameters():
                            p.requires_grad = True
                        self.optimizer.add_param_group(
                            {
                                "params": [
                                    *self.encoder.resnet.parameters(),
                                    *self.encoder.stats.parameters(),
                                    *self.encoder.fc2.parameters(),
                                ]
                            }
                        )

                    if self.encoder.use_step:
                        self.encoder.step(*self.training_point)

                    results_dict = self.fit_one_batch(s_batch, t_batch)

                    if self.encoder.criterion == "margin_loss":
                        results_dict[
                            "margin_percent"
                        ] = f"{1 / (1 + self.encoder.loss.lambda_factor):.2f}"

                    self.encoder.backward_step(*self.training_point)

                    # For multi-GPU training. Remember that it is not convenient to wrap lr_scheduler
                    # for there are many strategies with different details. Here, only warmR, ReduceLROnPlateau
                    # and some simple schedulers whose step() parameter is 'epoch' only are supported.
                    lr_scheduler_params = {"training_point": self.training_point}

                    self.valid_computed = False
                    if utils.is_main_training():
                        if self.data.valid_loader and self.reporter.is_report(self.training_point):
                            if not self.valid_computed:
                                (valid_metric, valid_results_dict,) = self.compute_validation(
                                    self.data.valid_loader
                                )
                                lr_scheduler_params["valid_metric"] = valid_metric
                                self.valid_computed = True

                            # real_snapshot is set for tensorboard to avoid workspace problem
                            real_snapshot = dict(**results_dict, **valid_results_dict)
                            snapshot = {
                                k: (
                                    "{0:.6f}".format(v)
                                    if "loss" in k
                                    else "{0:.2f}".format(v)
                                    if "acc" in k
                                    else v
                                )
                                for (k, v) in real_snapshot.items()
                            }
                            snapshot["real"] = {
                                k: v for (k, v) in real_snapshot.items() if type(v) == float
                            }

                        else:
                            real_snapshot = results_dict
                            snapshot = {
                                k: (
                                    "{0:.6f}".format(v)
                                    if "loss" in k
                                    else "{0:.2f}".format(v)
                                    if "acc" in k
                                    else v
                                )
                                for (k, v) in real_snapshot.items()
                            }
                            snapshot["real"] = {
                                k: v for (k, v) in real_snapshot.items() if type(v) == float
                            }

                    if self.lr_scheduler is not None:
                        self.step_lr_scheduler(lr_scheduler_params)

                    if utils.is_main_training():
                        self.reporter.update(snapshot)

                        if (
                            self.current_iter % self.saved_step == 0
                            and self.current_iter != 0
                            and not self.debug
                        ):
                            # self.save_ckpt 后面不要有随机化的操作
                            self.save_ckpt(recent_num_to_keep=100000)

                        if (
                            self.ckpt_interval_minutes > 0
                            and time.time() - last_ckpt_time >= self.ckpt_interval_minutes * 60.0
                            and not self.debug
                        ):
                            self.save_ckpt(recent_num_to_keep=1, recent_ckpt=True)
                            last_ckpt_time = time.time()

                # End of epoch
                self.resume_iter = 0

                # hasattr(self, "valid_computed"): "valid_computed" is not present when resuming
                # training. Skip saving checkpoint at the end of the epoch when resuming training.
                if utils.is_main_training() and not self.debug and hasattr(self, "valid_computed"):
                    if self.data.valid_loader is not None:
                        assert self.valid_computed is True
                        self.save_ckpt(
                            valid_loss=valid_metric[0],
                            valid_acc=valid_metric[1],
                            end_of_epoch=True,
                        )
                    else:
                        self.save_ckpt(recent_num_to_keep=50, end_of_epoch=True)

                # Initialize worker_state. Otherwise recover random state in worker_init_fn() will be re-executed.
                self.data.s_train_loader.dataset.worker_state = {}
                self.data.t_train_loader.dataset.worker_state = {}

            if utils.is_main_training():
                self.reporter.finish()
        except BaseException as e:
            if utils.use_ddp():
                utils.cleanup_ddp()
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()
            sys.exit(1)
