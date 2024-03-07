# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import yaml
import time
import math
import argparse
import traceback
from rich.progress import Progress
from hyperpyyaml import load_hyperpyyaml
from multiprocessing import Pool, Manager

speakernet = os.getenv("speakernet")
sys.path.insert(0, os.path.dirname(speakernet))

import speakernet.utils.utils as utils
from speakernet.utils.logging_utils import init_logger
from speakernet.utils.kaldi_common import StrToBoolAction
from speakernet.utils.rich_utils import progress_columns
from speakernet.pipelines.modules.extract_embs import extract

logger = init_logger()

# import warnings
# warnings.filterwarnings("ignore")

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
    description="""Train xvector framework with pytorch.""",
    formatter_class=argparse.RawTextHelpFormatter,
    conflict_handler="resolve",
)

parser.add_argument(
    "--gpu-id",
    type=str,
    default="0",
    help="set --gpu-id=1 2 3 to use multi-gpu to extract xvector.",
)
parser.add_argument(
    "--extract-epochs", type=str, default="19 20 21", help="xvector extracted epochs"
)
parser.add_argument("--train-time-string", type=str, default=" ")
parser.add_argument("--model-dir", type=str, default="", help="extract xvector dir name")
parser.add_argument("--jobs-per-gpu", type=int, default=3, help="")
parser.add_argument("--embedding-model", type=str, default="encoder.ckpt", help="")
parser.add_argument("--suffix", type=str, default="", help="outdir suffix")
parser.add_argument("--lower-epoch", type=str, default=0, help="")
parser.add_argument("--replacements", type=str, default="none", help="For replacing wav path.")

# Accept extra args to override yaml
args, overrides = parser.parse_known_args()
overrides_yaml = utils.convert_to_yaml(overrides)

time_string = args.train_time_string
logger.info(f"Timestamp: {time_string}")
model_dir = "exp/{}/{}".format(args.model_dir, time_string)
if not os.path.exists(model_dir):
    raise NotADirectoryError("time param error")

# Load hyperparameters file with command-line overrides
logger.info("Waiting for loading the hyperparams file ...")
while True:
    if not os.path.exists(f"{model_dir}/config/hyperparams.yaml"):
        time.sleep(10)
    else:
        break
with open(f"{model_dir}/config/hyperparams.yaml") as fin:
    hparams = load_hyperpyyaml(fin, overrides_yaml)

# >>> Set seed
utils.set_seed(hparams["seed"], deterministic=True)

# ---------------------------------------- START ------------------------------------------ #


def main():
    gpu_ids = args.gpu_id.split(" ")
    njobs = args.jobs_per_gpu * len(gpu_ids)

    # Define this w.r.t extracted_embedding param of encoder_blueprint.
    extract_positions = hparams["extract_positions"].split(" ")
    # All dataset should be in data_path.
    extract_data = hparams["extract_data"].split(" ")

    try:
        while True:
            if args.extract_epochs == "":
                while True:
                    extracted_embs = []
                    if os.path.exists(model_dir + "/embeddings") and not hparams["force"]:
                        for file in os.listdir(model_dir + "/embeddings"):
                            if "epoch" in file:
                                extracted = True
                                for extract_data_item in extract_data:
                                    if extract_data_item not in os.listdir(
                                        model_dir + f"/embeddings/{file}"
                                    ):
                                        extracted = False
                                if extracted:
                                    extracted_embs.append(file)

                    extract_embs = []
                    for checkpoint in os.listdir(model_dir + "/checkpoints"):
                        with open(f"{model_dir}/checkpoints/{checkpoint}/CKPT.yaml") as fi:
                            meta = yaml.load(fi, Loader=yaml.Loader)
                            if "recent-ckpt" in meta and meta["recent-ckpt"] is True:
                                continue
                            epoch = checkpoint.split("+")[1]
                            for position in extract_positions:
                                if f"{position}_epoch_{epoch}" not in extracted_embs:
                                    if (
                                        int(epoch.split('.')[0]) > int(args.lower_epoch.split('.')[0])
                                        or int(epoch.split('.')[0]) == int(args.lower_epoch.split('.')[0])
                                        and int(epoch.split('.')[1]) >= int(args.lower_epoch.split('.')[1])
                                    ):
                                        extract_embs.append([position, epoch])
                    # If run a trainer with background thread (do not be supported now) or run this launcher extrally with stage=1
                    # (it means another process), then this while-listen is useful to start extracting immediately (but require more gpu-memory).
                    if len(extract_embs) != 0:
                        break
                    else:
                        time.sleep(hparams["sleep_time"])
            else:
                # extract_epochs = ["19", "20", "21"]
                extract_epochs = args.extract_epochs.split(" ")
                extract_embs = [
                    [position, epoch] for epoch in extract_epochs for position in extract_positions
                ]

            ############## Sort ##############
            # Saved at the end of epoch cannot be split by '.', add fractional part manually
            extract_embs = [
                [position, str(int(epoch) + 1) + ".0"] if "." not in epoch else [position, epoch]
                for (position, epoch) in extract_embs
            ]
            extract_embs.sort(key=lambda x: (int(x[1].split(".")[0]), int(x[1].split(".")[1])))
            # Restore
            extract_embs = [
                [position, str(int(epoch.split(".")[0]) - 1)]
                if epoch.split(".")[1] == "0"
                else [position, epoch]
                for (position, epoch) in extract_embs
            ]

            # Run a batch extracting process.
            for (position, epoch) in extract_embs:
                extract_config = "{0}.extract.config".format(position)
                extract_config_path = f"{model_dir}/config/{extract_config}"
                if not os.path.exists(extract_config_path):
                    # Generate the extracting config from nnet config
                    # The position to extract depends on the 'extracted_embedding' parameter in encoder_creation (by my design).
                    encoder_blueprint, encoder_creation = utils.read_nnet_config(
                        "{0}/config/nnet.config".format(model_dir)
                    )
                    # To save memory without loading some independent components.
                    # string replace
                    encoder_creation = encoder_creation.replace("training=True", "training=False")
                    if "extracted_embedding" in hparams["encoder_params"]:
                        encoder_creation = encoder_creation.replace(
                            hparams["encoder_params"]["extracted_embedding"], position,
                        )
                    utils.write_nnet_config(
                        encoder_blueprint, encoder_creation, extract_config_path
                    )

                encoder_params = f"{model_dir}/checkpoints/epoch+{epoch}/{args.embedding_model}"

                # model_blueprint, model_creation = utils.read_nnet_config(extract_config_path, log=False)
                # model = utils.import_module(model_blueprint, model_creation)
                # model.load_state_dict(torch.load(encoder_params, map_location="cpu"), strict=False)
                # model.eval()

                # script_model = torch.jit.script(model)
                # script_model.save(f"{model_dir}/checkpoints/epoch+{epoch}/{args.embedding_model}")
                # logger.info(f'Export model successfully, see {f"{model_dir}/checkpoints/epoch+{epoch}/{args.embedding_model}"}')

                while True:
                    if os.path.exists(encoder_params):
                        break
                    else:
                        time.sleep(hparams["sleep_time"])

                for data in extract_data:
                    logger.info(f"Extracting {data} for epoch {epoch} in position {position} ...")
                    outdir = f"{model_dir}/embeddings/{position}_epoch_{epoch}/{data}{args.suffix}"
                    os.makedirs(outdir, exist_ok=True)

                    # Multiprocess pool
                    pool = Pool(njobs)
                    counter = Manager().Value(int, 0)
                    lock = Manager().Lock()

                    with open(f"{hparams['data_path']}/{data}/wav.scp", "r") as fr:
                        wav_paths = fr.readlines()
                    num_per_job = math.ceil(len(wav_paths) / njobs)

                    for job in range(njobs):
                        wavs_part = wav_paths[job * num_per_job : (job + 1) * num_per_job]
                        gpu_id = gpu_ids[job % len(gpu_ids)]
                        pool.apply_async(
                            extract,
                            args=(
                                wavs_part,
                                gpu_id,
                                job,
                                encoder_params,
                                extract_config_path,
                                outdir,
                                args.replacements,
                                counter,
                                lock,
                            ),
                        )

                    # Multiprocess progress bar
                    with Progress(*progress_columns) as progress:
                        task = progress.add_task("extracting...", total=len(wav_paths))
                        while True:
                            if counter.value > 0:
                                progress.update(task, completed=counter.value)
                            if counter.value == len(wav_paths):
                                break
                    pool.close()
                    pool.join()

                    # combining feats across jobs
                    scps = []
                    for job in range(njobs):
                        scps.extend(open(f"{outdir}/xvector.{job}.scp", "r").readlines())
                    with open(f"{outdir}/xvector.scp", "w") as scp_writer:
                        scp_writer.write("".join(scps))

            if args.extract_epochs == "":
                time.sleep(hparams["sleep_time"])
            else:
                break
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
