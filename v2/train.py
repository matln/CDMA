# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import yaml
import math
import torch
import shutil
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from hyperpyyaml import load_hyperpyyaml

# torch.multiprocessing.set_sharing_strategy('file_system')
# import torchaudio
# torchaudio.set_audio_backend("sox_io")

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

import speakernet.utils.utils as utils
from speakernet.training.optimizer import get_optimizer
from speakernet.training.checkpoints import Checkpointer
from speakernet.utils.kaldi_common import StrToBoolAction
from speakernet.training.lr_scheduler import LRSchedulerWrapper
from speakernet.utils.logging_utils import DispatchingFormatter, patch_logging_stream
from speakernet.utils.rich_utils import custom_console, MyRichHandler, MyReprHighlighter

warnings.filterwarnings("ignore")

# Logger
# Change the logging stream from stderr to stdout to be compatible with horovod.
patch_logging_stream(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = MyRichHandler(highlighter=MyReprHighlighter(), console=custom_console)
handler.setLevel(logging.INFO)
formatter = DispatchingFormatter(
    {"fit_progressbar": logging.Formatter("%(message)s", datefmt=" [%X]")},
    logging.Formatter("%(message)s", datefmt="[%X]"),
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
    description="""Train xvector framework with pytorch.""",
    formatter_class=argparse.RawTextHelpFormatter,
    conflict_handler="resolve",
)

parser.add_argument(
    "--hparams-file", type=str, help="A yaml-formatted file using the extended YAML syntax. ",
)
parser.add_argument(
    "--use-gpu",
    type=str,
    action=StrToBoolAction,
    default=True,
    choices=["true", "false"],
    help="Use GPU or not.",
)
parser.add_argument(
    "--gpu-id",
    type=str,
    default="",
    help="If NULL, then it will be auto-specified.\n"
    "set --gpu-id=1,2,3 to use multi-gpu to extract xvector.\n"
    "Doesn't support multi-gpu training",
)
parser.add_argument(
    "--multi-gpu-solution",
    type=str,
    default="ddp",
    choices=["ddp", "horovod", "dp"],
    help="if number of gpu_id > 1, this option will be valid to init a multi-gpu solution.",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=0,
    help="Do not delete it when using DDP-based multi-GPU training.\n"
    "It is important for torch.distributed.launch.",
)
parser.add_argument(
    "--port",
    type=int,
    default=29500,
    help="This port is used for DDP solution in multi-GPU training.",
)
parser.add_argument(
    "--debug", type=str, action=StrToBoolAction, default=False, choices=["true", "false"], help="",
)
parser.add_argument(
    "--mixed-prec",
    type=str,
    action=StrToBoolAction,
    default=False,
    choices=["true", "false"],
    help="",
)
parser.add_argument(
    "--resume-training",
    type=str,
    action=StrToBoolAction,
    default=False,
    choices=["true", "false"],
    help="",
)
parser.add_argument("--train-time-string", type=str, default=" ")
parser.add_argument("--model-dir", type=str, default="", help="extract xvector dir name")

# Accept extra args to override yaml
args, overrides = parser.parse_known_args()
overrides_yaml = utils.convert_to_yaml(overrides)

time_string = args.train_time_string
logger.info(f"Timestamp: {time_string}")
model_dir = "exp/{}/{}".format(args.model_dir, time_string)
description = ""

# >>> Init environment
# It is used for multi-gpu training if used (number of gpu-id > 1).
# And it will do nothing for single-GPU training.
utils.init_multi_gpu_training(args.gpu_id, args.multi_gpu_solution, args.port)

# Load hyperparameters file with command-line overrides
if args.resume_training:
    # Recover training
    with open(f"{model_dir}/config/hyperparams.yaml") as fin:
        hparams = load_hyperpyyaml(fin, overrides_yaml)
else:
    with open(args.hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides_yaml)

    if utils.is_main_training() and not args.debug:
        logger.info("Please provide a brief description for the experiment, or Enter nothing:")
        description = utils.input_with_timeout(timeout=300)

    # It is used for model.step() defined in model blueprint.
    if hparams["lr_scheduler_params"]["name"] == "warmR" and hparams["encoder_params"]["use_step"]:
        hparams["encoder_params"]["step_params"]["T"] = (
            hparams["lr_scheduler_params"]["warmR.T_max"],
            hparams["lr_scheduler_params"]["warmR.T_mult"],
        )
        overrides.append(
            "encoder_params.step_params.T={}".format(hparams["encoder_params"]["step_params"]["T"])
        )

    # Create experiment directory
    utils.create_model_dir(model_dir, args.debug)

    # Save all code files.
    utils.backup_code(args, hparams, model_dir, overrides, description)

# >>> Set seed
utils.set_seed(hparams["seed"], deterministic=True)

# ---------------------------------------- START ------------------------------------------ #
if utils.is_main_training():
    logger.info("Loading the dataset to a bunch.")
# The dict [info] contains feat_dim and num_targets.
bunch = utils.import_module(hparams["bunch"])
bunch, info = bunch.DataBunch.get_bunch_from_egsdir(
    *hparams["egs_dir"], hparams["dataset_params"], hparams["loader_params"]
)

if utils.is_main_training():
    logger.info("Loading the encoder.")
encoder = utils.import_module(hparams["encoder"])
encoder = encoder.Encoder(
    hparams["n_mels"],
    info["num_s_targets"],
    info["num_t_targets"],
    **hparams["encoder_params"],
    features=hparams["features"],
)

# If multi-GPU used, then batchnorm will be converted to synchronized batchnorm, which is important
# to make peformance stable.
# It will change nothing for single-GPU training.
encoder = utils.convert_synchronized_batchnorm(encoder)

encoder.load_transform_state_dict(torch.load(hparams["exist_encoder"], map_location="cpu"))
# print(encoder)
# print('------------------------')
# 
# convert_dabn(encoder)
# print(encoder)
# for p in encoder.resnet.parameters():
#     p.requires_grad = False
# for p in encoder.stats.parameters():
#     p.requires_grad = False
# for p in encoder.fc2.parameters():
#     p.requires_grad = False
# 
# print(len(list(filter(lambda p: p.requires_grad, encoder.parameters()))))

# Select device to GPU
# Order of distributedDataParallel(dataparallel) and optimizer has no effect
# https://medium.com/analytics-vidhya/distributed-training-in-pytorch-part-1-distributed-data-parallel-ae5c645e74cb
# https://discuss.pytorch.org/t/order-of-dataparallel-and-optimizer/114063
encoder = utils.select_model_device(
    encoder, args.use_gpu, gpu_id=args.gpu_id, benchmark=hparams["benchmark"]
)

if utils.is_main_training():
    logger.info("Define the optimizer.")
optimizer = get_optimizer(
    filter(lambda p: p.requires_grad, encoder.parameters()), hparams["optimizer_params"]
)
# for p in encoder.resnet.parameters():
#     p.requires_grad = True
# for p in encoder.stats.parameters():
#     p.requires_grad = True
# for p in encoder.fc2.parameters():
#     p.requires_grad = True
# optimizer.add_param_group(
#     {
#         "params": [
#             *encoder.resnet.parameters(),
#             *encoder.stats.parameters(),
#             *encoder.fc2.parameters(),
#         ]
#     }
# )

if utils.is_main_training():
    logger.info("Define the lr_scheduler.")
lr_scheduler = LRSchedulerWrapper(optimizer, hparams["lr_scheduler_params"])

if utils.is_main_training():
    logger.info("Define the checkpointer.")
recoverables = {
    "encoder": encoder,
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "s_dataloader": bunch.s_train_loader,
    "t_dataloader": bunch.t_train_loader,
}
checkpointer = Checkpointer(
    checkpoints_dir=f"{model_dir}/checkpoints", recoverables=recoverables, debug=args.debug,
)

if utils.is_main_training():
    logger.info("Initing the trainer.")
# Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
package = (
    {
        "data": bunch,
        "encoder": encoder,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "checkpointer": checkpointer,
    },
    {
        "model_dir": model_dir,
        "encoder_blueprint": f"{model_dir}/backup/{hparams['encoder']}",
        "exist_encoder": "",
        "gpu_id": args.gpu_id,
        "debug": args.debug,
        "report_times_every_epoch": hparams["report_times_every_epoch"],
        "ckpt_interval_minutes": hparams["ckpt_interval_minutes"],
        "epochs": hparams["epochs"],
        "report_interval_iters": hparams["report_interval_iters"],
        "record_file": "train.csv",
        "time_string": time_string,
        "mixed_prec": args.mixed_prec,
        "saved_step": hparams["saved_step"],
        "unfreeze_epoch": hparams["unfreeze_epoch"],
    },
)

trainer = utils.import_module(hparams["trainer"])
trainer = trainer.Trainer(package, stop_early=hparams["stop_early"])
trainer.fit()
