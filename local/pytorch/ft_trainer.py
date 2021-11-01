# -*- coding:utf-8 -*-

import os, sys
import math
import traceback
import pandas as pd
import numpy as np
import torch
from torch.cuda.amp import autocast
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from libs.training.trainer import SimpleTrainer, Reporter, LRSchedulerWrapper, logger
import libs.support.utils as utils
from .distance import compute_distance_matrix

"""
This is the structure of Package:

Package(
    Elements{
        data:Bunch
        model:TopVirtualNnet
        optimizer:Optimizer
        lr_scheduler:LR_Scheduler
        },

    Params{
        model_dir:str
        exist_model:str
        start_epoch:int
        epochs:int
        ...
        }
    )

trainer:
    self.elements
    self.params
    self.training_point(this_epoch, this_iter, data.num_batch_train)
"""

# Trainer ✿

class Trainer(SimpleTrainer):
    """One input and one output.
    """
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

    def train_one_batch(self, batch):
        """A normal training core without fetching data from iterator.
        """
        model = self.elements["model"]
        model_forward = self.elements["model_forward"]
        optimizer = self.elements["optimizer"]

        if not model.training:
            model.train()

        if self.params["nan_debug"]:
            device = utils.get_device(self.elements["model"])
            inputs = torch.load("{0}/nan.batch".format(self.params["model_dir"])).to(device)
            targets = torch.load("{0}/nan.targets".format(self.params["model_dir"])).to(device)
            self.elements["model"].load_state_dict(
                torch.load("{0}/nan.params".format(self.params["model_dir"]), map_location="cpu"))
            self.elements["model"].to(device)
        else:
            inputs, targets = batch
        optimizer.zero_grad()

        # --------------------------------------------------------------------------------------- #

        feats = model_forward(inputs)
        loss = model.get_t_loss(feats, targets)
        loss.backward()

        # --------------------------------------------------------------------------------------- #

        if self.params["max_change"] > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.params["max_change"])

            if math.isnan(grad_norm):
                if self.params["nan_debug"]:
                    raise RuntimeError("[NOT OK] Nan is still found in this debug.")
                torch.save(inputs.cpu(), "{0}/nan.batch".format(self.params["model_dir"]))
                torch.save(targets.cpu(), "{0}/nan.targets".format(self.params["model_dir"]))
                torch.save(self.elements["model"].state_dict(), "{0}/nan.params".format(self.params["model_dir"]))
                raise RuntimeError(
                    'There is Nan problem in iter/epoch: {0}/{1} '
                    '(nan batch and params are saved in {2})'.format(
                        self.training_point[1] + 1, self.training_point[0] + 1,
                        "{0}/nan.*".format(self.params["model_dir"])))
            else:
                if self.params["nan_debug"]:
                    raise RuntimeError("[OK] There is no nan found for this debug.")
                optimizer.step()
        else:
            optimizer.step()

        accuracy = model.get_t_accuracy(targets) if self.params["compute_accuracy"] else None

        return loss.detach().item(), accuracy


    def run(self):
        """Main function to start a training process.
        """
        try:
            self.init_training()

            if utils.is_main_training():
                self.reporter = Reporter(self)

            start_epoch = self.params["start_epoch"]
            epochs = self.params["epochs"]
            data = self.elements["data"]
            model = self.elements["model"]
            lr_scheduler = self.elements["lr_scheduler"]
            base_optimizer = self.elements["optimizer"]
            best_valid_acc = 0.0

            # For lookahead.
            if getattr(base_optimizer, "optimizer", None) is not None:
                base_optimizer = base_optimizer.optimizer
            last_lr = base_optimizer.state_dict()['param_groups'][0]['lr']

            if utils.is_main_training():
                logger.info("Training will run for {0} epochs.".format(epochs))

            for this_epoch in range(start_epoch, epochs):
                for this_iter, batch in enumerate(data.t_train_loader, 0):
                    self.training_point = (this_epoch, this_iter, data.num_batch_train)  # It is important for reporter.

                    if model.use_step:
                        model.step(*self.training_point)

                    loss_t_cls, acc = self.train_one_batch(batch)

                    model.backward_step(*self.training_point)

                    # For multi-GPU training. Remember that it is not convenient to wrap lr_scheduler 
                    # for there are many strategies with different details. Here, only warmR, ReduceLROnPlateau
                    # and some simple schedulers whose step() parameter is 'epoch' only are supported.
                    lr_scheduler_params = {"training_point": self.training_point}

                    valid_computed = False
                    if lr_scheduler.name == "reduceP" and lr_scheduler.is_reduce_point(self.training_point):
                        assert data.valid_loader is not None
                        valid_loss, valid_acc = self.compute_validation(data.valid_loader)
                        lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
                        valid_computed = True

                    if utils.is_main_training():
                        if valid_computed or (data.valid_loader and self.reporter.is_report(self.training_point)):
                            if not valid_computed:
                                valid_loss, valid_acc = self.compute_validation(data.valid_loader)
                                valid_computed = False

                            # real_snapshot is set for tensorboard to avoid workspace problem
                            real_snapshot = {"loss_t_cls": loss_t_cls, "valid_loss": valid_loss,
                                             "train_acc": acc * 100, "valid_acc": valid_acc * 100}
                            snapshot = {"loss_t_cls": "{0:.6f}".format(loss_t_cls),
                                        "valid_loss": "{0:.6f}".format(valid_loss),
                                        "train_acc": "{0:.2f}".format(acc * 100),
                                        "valid_acc": "{0:.2f}".format(valid_acc * 100),
                                        "real": real_snapshot}
                            # For ReduceLROnPlateau.
                            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)

                            if lr_scheduler.name == "warmR":
                                if this_epoch >= epochs - 1 and valid_acc >= best_valid_acc:
                                    best_valid_acc = valid_acc
                                    self.save_model(from_epoch=False)
                        else:
                            real_snapshot = {"loss_t_cls": loss_t_cls, "train_acc": acc * 100}
                            snapshot = {"loss_t_cls": "{0:.6f}".format(loss_t_cls),
                                        "valid_loss": "",
                                        "train_acc": "{0:.2f}".format(acc * 100), "valid_acc": "",
                                        "real": real_snapshot}

                    if lr_scheduler is not None:
                        # It is not convenient to wrap lr_scheduler (doing).
                        if isinstance(lr_scheduler, LRSchedulerWrapper):
                            lr_scheduler.step(**lr_scheduler_params)
                            if utils.is_main_training():
                                current_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
                                if lr_scheduler.name == "reduceP":
                                    if current_lr < last_lr:
                                        last_lr = current_lr
                                        self.save_model(from_epoch=False)
                                    elif current_lr <= lr_scheduler.min_lr and lr_scheduler.is_reduce_point(self.training_point):
                                        self.save_model(from_epoch=False)
                                elif lr_scheduler.name == "cyclic" and utils.is_main_training():
                                    cyclic_size = lr_scheduler.lr_scheduler.total_size
                                    current_iter = self.training_point[0] * self.training_point[2] + self.training_point[1] + 1
                                    if current_iter % cyclic_size == 0 and current_iter != 1:
                                        self.save_model(from_epoch=False)
                        else:
                            # For some pytorch lr_schedulers, but it is not available for all.
                            lr_scheduler.step(this_epoch)
                    if utils.is_main_training():
                        self.reporter.update(snapshot)

                if utils.is_main_training():
                    self.save_model()

            if utils.is_main_training():
                self.reporter.finish()
        except BaseException as e:
            if utils.use_ddp():
                utils.cleanup_ddp()
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    pass
