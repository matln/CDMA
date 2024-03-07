# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import copy
import random
import logging
import argparse
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

from speakernet.utils.utils import set_seed
from speakernet.utils.kaldi_common import StrToBoolAction
from speakernet.utils.logging_utils import DispatchingFormatter
from speakernet.pipelines.modules.kaldi_dataset import KaldiDataset
from speakernet.utils.rich_utils import (
    track,
    MyRichHandler,
)

from samples import WaveformSamples


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = MyRichHandler()
handler.setLevel(logging.INFO)
formatter = DispatchingFormatter(
    {'fit_progressbar': logging.Formatter("%(message)s", datefmt=" [%X]")},
    logging.Formatter("%(message)s", datefmt="[%X]"))
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    # Start
    parser = argparse.ArgumentParser(
        description="""Split data to chunk-egs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )

    # Options
    # valid: validation set
    parser.add_argument(
        "--duration", type=float, default=2.0, help="duration of chunks"
    )

    parser.add_argument("--min-duration", type=float, default=2.0, help="")

    parser.add_argument(
        "--sample-type",
        type=str,
        default="speaker_balance",
        choices=["speaker_balance", "sequential", "sequential_with_domain", "every_utt", "random_segment"],
        help="The sample type for trainset.",
    )

    parser.add_argument(
        "--chunk-num",
        type=int,
        default=-1,
        help="Define the avg chunk num. -1->suggestion"
        "（total_num_chunks / num_spks * scale）, 0->max_num_chunks, int->int",
    )

    parser.add_argument(
        "--scale", type=float, default=1.5, help="The scale for --chunk-num:-1."
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="The scale of overlap to generate chunks.",
    )
    parser.add_argument("--frame-overlap", type=float, default=0.015, help="")

    parser.add_argument(
        "--drop-last-duration",
        type=float,
        default=0.2,
        help="If the duration of the last chunk < drop-last-duration,"
        " it will be dropped.",
    )

    parser.add_argument("--valid-dir", type=str, default="", help="A kaldi datadir.")

    parser.add_argument(
        "--valid-split-from-trainset",
        type=str,
        action=StrToBoolAction,
        default=True,
        choices=["true", "false"],
        help="",
    )

    parser.add_argument(
        "--valid-split-type",
        type=str,
        default="--total-spk",
        choices=["--default", "--per-spk", "--total-spk"],
        help="Get the valid samples or not.",
    )

    parser.add_argument(
        "--valid-num-spks",
        type=int,
        default=1024,
        help="The num spks to split for valid. 1024 for --total-spk.",
    )

    parser.add_argument(
        "--valid-chunk-num",
        type=int,
        default=2,
        help="define the avg chunk num. -1->suggestion"
        "（max / num_spks * scale）, 0->max, int->int"
        "chunk num of every validation set utterance ",
    )

    parser.add_argument(
        "--valid-sample-type",
        type=str,
        default="every_utt",
        choices=["speaker_balance", "sequential", "every_utt", "full_length"],
        help="The sample type for valid set.",
    )

    parser.add_argument(
        "--valid-scale",
        type=float,
        default=1.5,
        help="The scale for --valid-chunk-num:-1.",
    )

    parser.add_argument("--seed", type=int, default=1024, help="random seed")

    parser.add_argument("--expected-files", type=str, default="utt2spk,spk2utt,wav.scp")

    parser.add_argument(
        "--amp-th",
        type=float,
        default=0.0005,
        help="removes segments whose average amplitude is below the" "given threshold.",
    )

    parser.add_argument("--samplerate", type=int, default=16000, help="")

    # Main
    parser.add_argument(
        "data_dir", metavar="data-dir", type=str, help="A kaldi datadir."
    )
    parser.add_argument(
        "save_dir",
        metavar="save-dir",
        type=str,
        help="The save dir of mapping file of chunk-egs.",
    )

    # End
    print(" ".join(sys.argv))
    args = parser.parse_args()

    return args


def get_waveform_chunk_egs(args):
    logger.info("Loading kaldi datadir {0}".format(args.data_dir))
    expected_files = args.expected_files.split(",")
    if "feats.scp" in expected_files or "utt2num_frames" in expected_files:
        raise ValueError
    trainset = KaldiDataset.load_data_dir(args.data_dir, expected_files=expected_files, wav_scp_vector=True)

    if "utt2spk_int" not in expected_files:
        trainset.generate("utt2spk_int")
    if "utt2domain" in expected_files and "domain2utt" in expected_files:
        trainset.generate("utt2domain_int")

    logger.info("Generating chunk egs with duration={0}.".format(args.duration))
    # 按照一定的overlap将每个utterance分割成chunk
    trainset_samples = WaveformSamples(
        trainset,
        args.duration,
        chunk_type=args.sample_type,
        chunk_num_selection=args.chunk_num,
        scale=args.scale,
        overlap=args.overlap,
        frame_overlap=args.frame_overlap,
        drop_last_duration=args.drop_last_duration,
        samplerate=args.samplerate,
        min_duration=args.min_duration,
        amp_th=args.amp_th,
    )

    if trainset_samples.skipped_utts != []:
        old_spk2utt = copy.deepcopy(trainset.spk2utt)
        trainset = trainset.filter(trainset_samples.skipped_utts, id_type="utt", exclude=True)
        if len(old_spk2utt) != len(trainset.spk2utt):
            excluded_spks = list(set(list(old_spk2utt.keys())) - set(list(trainset.spk2utt.keys())))
            logger.error(f"utts of {','.join(excluded_spks)} are all silent.")

    if args.valid_dir != "":
        valid = KaldiDataset.load_data_dir(
            args.valid_dir, expected_files=expected_files, wav_scp_vector=True
        )
        if "utt2spk_int" not in expected_files:
            valid.generate("utt2spk_int", trainset.spk2int)

        valid_sample = WaveformSamples(
            valid,
            args.duration,
            chunk_type=args.valid_sample_type,
            chunk_num_selection=args.valid_chunk_num,
            scale=args.valid_scale,
            overlap=args.overlap,
            frame_overlap=args.frame_overlap,
            drop_last_duration=args.drop_last_duration,
            samplerate=args.samplerate,
            min_duration=args.min_duration,
            amp_th=args.amp_th,
        )

    elif args.valid_split_from_trainset:
        logger.info("Split valid dataset from {0}".format(args.data_dir))
        valid_num_spks = min(args.valid_num_spks, trainset.num_spks)

        if (
            valid_num_spks * args.valid_chunk_num
            > len(trainset_samples.chunk_samples) // 10
        ):
            logger.info(
                "Warning: the num of valid chunks ({0}) is out of 1/10 * num of dataset chunks ({1})."
                " Where --valid-num-spks={2} --valid-chunk-num={3}. Suggest to be less.".format(
                    valid_num_spks * args.valid_chunk_num,
                    len(trainset_samples.chunk_samples),
                    valid_num_spks,
                    args.valid_chunk_num,
                )
            )

        logger.info("Select valset chunks.")
        valid_sample = copy.deepcopy(trainset_samples)
        valid_chunks = []
        if args.valid_split_type == "--total-spk":
            logger.info("Select valset chunks with --total-spk requirement")
            spks = list(
                np.random.choice(
                    list(trainset.spk2utt.keys()), valid_num_spks, replace=False
                )
            )
        else:
            raise NotImplementedError

        for spk in spks:
            valid_chunks.extend(
                random.sample(trainset_samples.spk2chunks[spk], args.valid_chunk_num)
            )
        valid_chunks = sorted(valid_chunks, key=lambda x: x[6])
        valid_sample.chunk_samples = valid_chunks

        logger.info("Exclude valset chunks from trainset chunks.")
        valid_chunks_context = {}
        for item in valid_chunks:
            utt, counter = item[0].rsplit("-", 1)
            spk = utt.split("-")[0]
            valid_chunks_context.setdefault(spk, []).append(item[0])
            if args.overlap > 0.0:
                valid_chunks_context.setdefault(spk, []).append(
                    utt + f"-{int(counter)-1}"
                )
                valid_chunks_context.setdefault(spk, []).append(
                    utt + f"-{int(counter)+1}"
                )
        _chunk_samples = []
        for item in track(trainset_samples.chunk_samples):
            if item[0].split("-")[0] in valid_chunks_context:
                if item[0] not in valid_chunks_context[item[0].split("-")[0]]:
                    _chunk_samples.append(item)
            else:
                _chunk_samples.append(item)
        trainset_samples.chunk_samples = _chunk_samples

    logger.info(f"Skipped {len(trainset_samples.skipped_utts)} utterances.")

    logger.info("Save mapping file of chunk egs to {0}".format(args.save_dir))
    trainset_samples.save("{0}/train.egs.csv".format(args.save_dir))

    logger.info("Save mapping file of chunk egs to {0}".format(args.save_dir))
    if not os.path.exists("{0}/info".format(args.save_dir)):
        os.makedirs("{0}/info".format(args.save_dir))

    if args.valid_dir != "" or args.valid_split_from_trainset is True:
        valid_sample.save("{0}/validation.egs.csv".format(args.save_dir))

    with open("{0}/info/num_targets".format(args.save_dir), "w") as writer:
        writer.write(str(trainset.num_spks))

    if hasattr(trainset, "domain2utt"):
        with open("{0}/info/num_domain_targets".format(args.save_dir), "w") as writer:
            writer.write(str(len(trainset.domain2utt)))

    logger.info("Generate egs from {0} done.".format(args.data_dir))


def main():
    args = get_args()

    set_seed(args.seed)

    try:
        get_waveform_chunk_egs(args)
    except BaseException as e:
        # Look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
