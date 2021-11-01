# -*- coding:utf-8 -*-

import sys
import os
import logging
import argparse
import traceback
import time
import math
import numpy as np
import pandas as pd
import shutil

import torch

subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))

import local.pytorch.egs as egs
import libs.training.optim as optim
import libs.training.lr_scheduler as learn_rate_scheduler
import libs.support.kaldi_common as kaldi_common
import libs.support.utils as utils
from libs.support.logging_stdout import patch_logging_stream
import warnings
warnings.filterwarnings("ignore")


# Logger
# Change the logging stream from stderr to stdout to be compatible with horovod.
patch_logging_stream(logging.INFO)

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
    description="""Train xvector framework with pytorch.""",
    formatter_class=argparse.RawTextHelpFormatter,
    conflict_handler='resolve')

parser.add_argument("--model-dir", type=str, default="", help="extract xvector dir name")
parser.add_argument("--stage", type=int, default=0,
                    help="The stage to control the start of training epoch (default 0).\n"
                         "    stage 0: training.\n"
                         "    stage 1: extract xvector.")
parser.add_argument("--train-stage", type=int, default=-1,
                    help="The stage to control the start of training epoch (default -1).\n"
                         "    -1 -> creating model_dir.\n"
                         "     0 -> model initialization (e.g. transfer learning).\n"
                         "    >0 -> recovering training.")
parser.add_argument("--use-gpu", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Use GPU or not.")
parser.add_argument("--gpu-id", type=str, default="",
                    help="If NULL, then it will be auto-specified.\n"
                         "set --gpu-id=1,2,3 to use multi-gpu to extract xvector.\n"
                         "Doesn't support multi-gpu training")
parser.add_argument("--port", type=int, default=29500,
                    help="This port is used for DDP solution in multi-GPU training.")
parser.add_argument("--benchmark", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="If true, save training time but require a little more gpu-memory.")
parser.add_argument("--run-lr-finder", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="If true, run lr finder rather than training.")
parser.add_argument("--seed", type=int, default=1024,
                    help="random seed")
parser.add_argument("--source-egs-dir", type=str,
                    default="exp/egs/fbank64-voxceleb1o2_train-400-sequential-novad-ssd",
                    help="Network input chunks directory")
parser.add_argument("--target-egs-dir", type=str, default=None,
                    help="Network input chunks directory")
parser.add_argument("--train-csv-name", type=str, default="train.egs.csv", help="")

parser.add_argument("--debug", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"], help="")
parser.add_argument("--mode", type=str, default="pretrain",
                    choices=["pretrain", "finetuning", "SDA", "UDA", "emb_mmd"],
                    help="SDA: supervised domain adaptation"
                         "UDA: unsupervised domain adaptation"
                         "emb_mmd: calculate mmd loss on embedding features")

parser.add_argument("--extract-positions", type=str, default="far,near",
                    help="xvector extracted positions")
parser.add_argument("--extract-data", type=str, default="train_mix,test_clean,test_all_noises",
                    help="xvector extracted data")
parser.add_argument("--extract-epochs", type=str, default="19,20,21",
                    help="xvector extracted epochs")
parser.add_argument("--feature", type=str, default="fbank_40_pitch", help="")


parser.add_argument("--use-fast-loader", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"], help="")
parser.add_argument("--batch-size", type=int, default=512, help="")
parser.add_argument("--learn-rate", type=float, default=0.04, help="")
parser.add_argument("--margin-loss", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"], help="")
parser.add_argument("--use-step", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"], help="")
parser.add_argument("--t-margin-loss", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"], help="")
parser.add_argument("--t-use-step", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"], help="")
parser.add_argument("--lr-scheduler", type=str, default="reduceP",
                    choices=["reduceP", "stepLR", "MultiStepLR"], help="")
parser.add_argument("--milestones", type=str, default="1,2", help="")
parser.add_argument("--gamma1", type=float, default=0.1, help="")
parser.add_argument("--step-size", type=int, default=3, help="")
parser.add_argument("--gamma2", type=float, default=0.1, help="")
parser.add_argument("--num-chunks", type=int, default=4, help="")


parser.add_argument("--epochs", type=int, default=30, help="")
parser.add_argument("--exist-model", type=str, default="", help="")

parser.add_argument("--train-time-string", type=str, default=" ")
parser.add_argument("--mixed-prec", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"], help="")

args = parser.parse_args()

######################################### PARAMS ####################################################

# --------------------------------------------------#
# Control options
train_stage = max(-1, args.train_stage)

# --------------------------------------------------#
# Training options
egs_params = {
    # SpecAugment.
    # None or specaugment. If use aug, you should close the aug_dropout which is in model_params.
    "aug": None,
    "aug_params": {"frequency": 0.2, "frame": 0.2, "rows": 4, "cols": 4, "random_rows": True, "random_rows": True},
    "egs_type": "chunk",
    "s_train_csv_name": args.train_csv_name,
    "chunk_size": int(args.source_egs_dir.split('-')[2])
}

loader_params = {
    # It is a queue loader to prefetch batch and storage.
    "use_fast_loader": args.use_fast_loader,
    "max_prefetch": 10,
    "batch_size": args.batch_size,
    "num_workers": 0,
    "pin_memory": False,
    "num_chunks": args.num_chunks,
}

# Difine model_params by model_blueprint w.r.t your model's __init__(model_params).
model_params = {
    "aug_dropout": 0., "tail_dropout": 0.,
    "training": True, "extracted_embedding": "far",
    "resnet_params": {
        "head_conv": True, "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
        "head_maxpool": False, "head_maxpool_params": {"kernel_size": 3, "stride": 2, "padding": 1},
        "block": "BasicBlock",
        "layers": [3, 4, 6, 3],
        "planes": [32, 64, 128, 256],
        "convXd": 2,
        "norm_layer_params": {"momentum": 0.5, "affine": True},
        "full_pre_activation": False,
        "zero_init_residual": False},

    "pooling": "statistics",  # statistics, lde, attentive, multi-head, multi-resolution
    "pooling_params":{
        "num_head":16,
        "share":True,
        "affine_layers":1,
        "hidden_size":64,
        "context":[0],
        "stddev":True,
        "temperature":True, 
        "fixed":True},

    "fc1": True,
    "fc1_params": {
        "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

    "fc2_params": {
        "nonlinearity": '', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

    "margin_loss": True if args.margin_loss else False,
    "margin_loss_params": {
        "method": "am", "m": 0.2, "feature_normalize": True,
        "s": 30, "mhe_loss": False, "mhe_w": 0.01},

    "use_step": True if args.use_step else False,
    "step_params": {
        "T": None,
        "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
        "s": False, "s_tuple": (30, 12), "s_list": None,
        "t": False, "t_tuple": (0.5, 1.2),
        "p": False, "p_tuple": (0.5, 0.1)},

    "t_margin_loss": True if args.t_margin_loss else False,
    "t_margin_loss_params": {
        "method": "am", "m": 0.2, "feature_normalize": True,
        "s": 30, "mhe_loss": False, "mhe_w": 0.01},

    "t_use_step": True if args.t_use_step else False,
    "t_step_params": {
        "T": None,
        "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
        "s": False, "s_tuple": (30, 12), "s_list": None,
        "t": False, "t_tuple": (0.5, 1.2),
        "p": False, "p_tuple": (0.5, 0.1)}
}

optimizer_params = {
    "name": "sgd",
    "learn_rate": args.learn_rate,
    "beta1": 0.9,
    "weight_decay": 5e-4,
}

lr_scheduler_params = {
    "name": args.lr_scheduler,
    "MultiStepLR.milestones": [int(x) for x in args.milestones.split(',')],
    "MultiStepLR.gamma": args.gamma1,
    "stepLR.step_size": args.step_size,
    "stepLR.gamma": args.gamma2,
    "reduceP.metric": 'valid_loss',
    "reduceP.check_interval": 8000,  # 0 means check metric after every epoch and 1 means every iter.
    "reduceP.factor": 0.1,  # scale of lr in every times.
    "reduceP.patience": 2,
    "reduceP.threshold": 0.0001,
    "reduceP.cooldown": 0,
    "reduceP.min_lr": 1e-8
}

epochs = args.epochs    # Total epochs to train. It is important.

report_times_every_epoch = None
# About validation computation and loss reporting. If report_times_every_epoch is not None,
if args.mode == "pretrain":
    report_interval_iters = 1000
else:
    report_interval_iters = 100
# then compute report_interval_iters by report_times_every_epoch.
stop_early = False
suffix = "params"    # Used in saved model file.
# -------------------------------------------------- #
# Other options
exist_model = args.exist_model  # Use it in transfer learning.
# -------------------------------------------------- #

# ---------------------------------------- START ------------------------------------------ #

# >>> Set seed
utils.set_all_seed(args.seed, deterministic=True)

# >>> Train model
if args.stage == 0:
    time_string = args.train_time_string
    # -------------------------------------------------- #
    # Main params
    model_dir = "exp/{}/{}".format(args.model_dir, time_string)
    if utils.is_main_training():
        if os.path.exists(model_dir):
            print(model_dir)
            raise ValueError("time param error")


    if utils.is_main_training():
        if args.debug is False:
            shutil.copytree("local/pytorch", os.path.join(model_dir, "config/pytorch"))
            model_blueprint = os.path.join(model_dir, "config/pytorch/resnet-xvector.py")
        else:
            model_blueprint = "local/pytorch/resnet-xvector.py"


    if utils.is_main_training(): logger.info("Load egs to bunch.")
    # The dict [info] contains feat_dim and num_targets.
    bunch, info = egs.Bunch.get_bunch_from_egsdir(args.source_egs_dir, args.target_egs_dir,
                                                  egs_params=egs_params, data_loader_params_dict=loader_params)


    if utils.is_main_training(): logger.info("Create model from model blueprint.")
    # Another way: import the model.py in this python directly, but it is not friendly to the shell script of extracting and
    # I don't want to change anything about extracting script when the model.py is changed.
    # 动态导入模块就是只知道str类型的模块名字符串，通过这个字符串导入模块
    model_py = utils.create_model_from_py(model_blueprint)
    model = model_py.ResNetXvector(info["feat_dim"], info["num_targets"], info["t_num_targets"],
                                   **model_params,
                                   batch_size=args.batch_size,
                                   num_chunks=args.num_chunks,
                                   mode=args.mode)


    if utils.is_main_training(): logger.info("Define optimizer and lr_scheduler.")
    optimizer = optim.get_optimizer(model, optimizer_params)
    lr_scheduler = learn_rate_scheduler.LRSchedulerWrapper(optimizer, lr_scheduler_params)


    # Record params to model_dir
    if args.debug is False:
        utils.write_list_to_file([args, egs_params, loader_params, model_params, optimizer_params, lr_scheduler_params],
                                 model_dir + '/config/params.dict')


    if utils.is_main_training(): logger.info("Init a trainer.")
    # Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
    package = ({"data": bunch, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler},
               {"model_dir": model_dir, "model_blueprint": model_blueprint, "exist_model": exist_model,
                "start_epoch": train_stage, "epochs": epochs, "use_gpu": args.use_gpu, "gpu_id": args.gpu_id,
                "benchmark": args.benchmark, "suffix": suffix, "report_times_every_epoch": report_times_every_epoch,
                "report_interval_iters": report_interval_iters, "debug": args.debug,
                "record_file": "train.csv", "time_string": time_string, "mixed_prec": args.mixed_prec})

    if args.mode == "SDA":
        import local.pytorch.sda_trainer as trainer
        trainer = trainer.Trainer(package, stop_early=stop_early)
    elif args.mode == "UDA":
        import local.pytorch.uda_trainer as trainer
        trainer = trainer.Trainer(package, stop_early=stop_early)
    elif args.mode == "emb_mmd":
        import local.pytorch.emb_mmd_trainer as trainer
        trainer = trainer.Trainer(package, stop_early=stop_early)
    elif args.mode == "finetuning":
        import local.pytorch.ft_trainer as trainer
        trainer = trainer.Trainer(package, stop_early=stop_early)
    else:
        import libs.training.trainer as trainer
        trainer = trainer.SimpleTrainer(package, stop_early=stop_early)

    if args.run_lr_finder and utils.is_main_training():
        trainer.run_lr_finder("lr_finder.csv", init_lr=1e-8, final_lr=10., num_iters=2000, beta=0.98)
        endstage = 0  # Do not start extractor.
    else:
        trainer.run()


# Extract xvector
if args.stage == 1 and utils.is_main_training():
    # There are some params for xvector extracting.
    data_root = "data"  # It contains all dataset just like Kaldi recipe.
    prefix = args.feature  # For to_extracted_data.

    nj = 4 * len(args.gpu_id.split(','))
    # nj = 32
    force = True
    use_gpu = True
    sleep_time = 10
    cmn = True

    time_string = args.train_time_string
    print(time_string)
    model_dir = "exp/{}/{}".format(args.model_dir, time_string)
    if not os.path.exists(model_dir):
        raise ValueError("time param error")

    # Define this w.r.t extracted_embedding param of model_blueprint.
    to_extracted_positions = args.extract_positions.split(',')
    # All dataset should be in data_root/prefix.
    to_extracted_data = args.extract_data.split(',')
    # It is model's name, such as 10.params or final.params (suffix is w.r.t package).
    if args.extract_epochs == "":
        saved_params = os.listdir(model_dir + "/params")
        to_extracted_epochs = [item.split('.')[0] for item in saved_params]
        to_extracted_epochs.sort()
    else:
        # to_extracted_epochs = ["19", "20", "21"]
        to_extracted_epochs = args.extract_epochs.split(',')

    # write the extracted epochs to a config file 
    with open(model_dir + "/config/extracted_epochs", 'w') as writer:
        writer.write(' '.join(to_extracted_epochs))

    # Run a batch extracting process.
    try:
        for position in to_extracted_positions:
            # Generate the extracting config from nnet config where
            # which position to extract depends on the 'extracted_embedding' parameter of model_creation (by my design).
            model_blueprint, model_creation = utils.read_nnet_config(
                "{0}/config/nnet.config".format(model_dir))
            # To save memory without loading some independent components.
            # string replace
            model_creation = model_creation.replace("training=True", "training=False")
            model_creation = model_creation.replace(model_params["extracted_embedding"], position)
            extract_config = "{0}.extract.config".format(position)
            if not args.debug:
                utils.write_nnet_config(model_blueprint, model_creation,
                                        "{0}/config/{1}".format(model_dir, extract_config))
            for epoch in to_extracted_epochs:
                model_file = "{0}/{1}.{2}".format(suffix, epoch, suffix)
                point_name = "{0}_epoch_{1}".format(position, epoch)

                # If run a trainer with background thread (do not be supported now) or run this launcher extrally with stage=1
                # (it means another process), then this while-listen is useful to start extracting immediately (but require more gpu-memory).
                model_path = "{0}/{1}".format(model_dir, model_file)
                while True:
                    if os.path.exists(model_path):
                        break
                    else:
                        time.sleep(sleep_time)

                for data in to_extracted_data:
                    datadir = "{0}/{1}/{2}".format(data_root, prefix, data)
                    outdir = "{0}/{1}/{2}".format(model_dir, point_name, data)
                    # Use a well-optimized shell script (with multi-processes) to extract xvectors.
                    # Another way: use subtools/splitDataByLength.sh and subtools/pytorch/pipeline/onestep/extract_embeddings.py
                    # with python's threads to extract xvectors directly, but the shell script is more convenient.
                    kaldi_common.execute_command(
                        "bash {subtools}/pytorch/pipeline/extract_xvectors_for_pytorch.sh "
                        "--model {model_file} --cmn {cmn} --nj {nj} --use-gpu {use_gpu} --gpu-id '{gpu_id}' "
                        " --force {force} --nnet-config config/{extract_config} --preprocess false "
                        "{model_dir} {datadir} {outdir}".format(
                            subtools=subtools, model_file=model_file, cmn=str(cmn).lower(), nj=nj,
                            use_gpu=str(use_gpu).lower(), gpu_id=args.gpu_id, force=str(force).lower(),
                            extract_config=extract_config,
                            model_dir=model_dir, datadir=datadir, outdir=outdir))
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)
