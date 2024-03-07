# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import glob
import shutil
import logging
import argparse
import torchaudio
import pandas as pd

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

from speakernet.utils.kaldi_common import StrToBoolAction
from speakernet.utils.logging_utils import DispatchingFormatter
from speakernet.utils.rich_utils import (
    track,
    custom_console,
    MyRichHandler,
    MyReprHighlighter,
)


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


def prepare_speech_aug(
    openrir_folder,
    musan_folder,
    csv_folder="exp/aug_csv",
    save_folder="data/speech_aug2",
    max_noise_len=2.015,
    overlap=0,
    force_clear=False,
):
    """Prepare the openrir and musan dataset for adding reverb and noises.

    Arguments
    ---------
    openrir_folder,musan_folder : str
        The location of the folder containing the dataset.
    csv_folder : str
        csv file save dir.
    save_folder : str
        Dir for saving the processed noise wav.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    overlap : float
    force_clear : bool
        whether clear the old dir.
    """

    if not os.path.isdir(os.path.join(openrir_folder, "RIRS_NOISES")):
        raise OSError(
            "{} is not exist, please download it.".format(
                os.path.join(openrir_folder, "RIRS_NOISES")
            )
        )
    if not os.path.isdir(os.path.join(musan_folder, "musan")):
        raise OSError(
            "{} is not exist, please download it.".format(
                os.path.join(musan_folder, "musan")
            )
        )

    if force_clear:
        if os.path.isdir(csv_folder):
            shutil.rmtree(csv_folder)
        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)

    if not os.path.isdir(csv_folder):
        os.makedirs(csv_folder)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    musan_speech_files = glob.glob(os.path.join(musan_folder, "musan/speech/*/*.wav"))
    musan_music_files = glob.glob(os.path.join(musan_folder, "musan/music/*/*.wav"))
    musan_noise_files = glob.glob(os.path.join(musan_folder, "musan/noise/*/*.wav"))

    musan_speech_item = []
    musan_music_item = []
    musan_noise_item = []
    for file in musan_speech_files:
        new_filename = os.path.join(save_folder, "/".join(file.split("/")[-4:]))
        musan_speech_item.append((file, new_filename))
    for file in musan_music_files:
        new_filename = os.path.join(save_folder, "/".join(file.split("/")[-4:]))
        musan_music_item.append((file, new_filename))
    for file in musan_noise_files:
        new_filename = os.path.join(save_folder, "/".join(file.split("/")[-4:]))
        musan_noise_item.append((file, new_filename))

    rir_point_noise_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
    )
    rir_point_noise_item = []
    for line in open(rir_point_noise_filelist):
        file = line.split()[-1]
        file_name = os.path.join(openrir_folder, file)
        new_filename = os.path.join(save_folder, file)
        rir_point_noise_item.append((file_name, new_filename))

    real_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
    )
    real_rir_rev_item = []
    for line in open(real_rir_filelist):
        file = line.split()[-1]
        file_name = os.path.join(openrir_folder, file)
        new_filename = os.path.join(save_folder, file)
        real_rir_rev_item.append((file_name, new_filename))

    sim_medium_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "simulated_rirs", "mediumroom", "rir_list"
    )
    sim_medium_rir_rev_item = []
    for line in open(sim_medium_rir_filelist):
        file = line.split()[-1]
        file_name = os.path.join(openrir_folder, file)
        file_dir, base_file = file.rsplit("/", 1)
        new_base_file = "medium_" + base_file
        new_filename = os.path.join(save_folder, file_dir, new_base_file)
        sim_medium_rir_rev_item.append((file_name, new_filename))

    sim_small_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "simulated_rirs", "smallroom", "rir_list"
    )
    sim_small_rir_rev_item = []
    for line in open(sim_small_rir_filelist):
        file = line.split()[-1]
        file_name = os.path.join(openrir_folder, file)
        file_dir, base_file = file.rsplit("/", 1)
        new_base_file = "small_" + base_file
        new_filename = os.path.join(save_folder, file_dir, new_base_file)
        sim_small_rir_rev_item.append((file_name, new_filename))
    sim_large_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "simulated_rirs", "largeroom", "rir_list"
    )
    sim_large_rir_rev_item = []
    for line in open(sim_large_rir_filelist):
        file = line.split()[-1]
        file_name = os.path.join(openrir_folder, file)
        file_dir, base_file = file.rsplit("/", 1)
        new_base_file = "large_" + base_file
        new_filename = os.path.join(save_folder, file_dir, new_base_file)
        sim_large_rir_rev_item.append((file_name, new_filename))

    noise_items = musan_noise_item
    csv_dct = {}
    reverb_csv = os.path.join(csv_folder, "real_reverb.csv")
    csv_dct[reverb_csv] = real_rir_rev_item
    sim_small_csv = os.path.join(csv_folder, "sim_small_reverb.csv")
    csv_dct[sim_small_csv] = sim_small_rir_rev_item
    sim_medium_csv = os.path.join(csv_folder, "sim_medium_reverb.csv")
    csv_dct[sim_medium_csv] = sim_medium_rir_rev_item
    sim_large_csv = os.path.join(csv_folder, "sim_large_reverb.csv")
    csv_dct[sim_large_csv] = sim_large_rir_rev_item
    noise_csv = os.path.join(csv_folder, "musan_noise.csv")
    pointsrc_noises_csv = os.path.join(csv_folder, "pointsrc_noise.csv")
    csv_dct[pointsrc_noises_csv] = rir_point_noise_item
    noise_csv = os.path.join(csv_folder, "musan_noise.csv")
    csv_dct[noise_csv] = noise_items
    bg_music_csv = os.path.join(csv_folder, "musan_music.csv")
    csv_dct[bg_music_csv] = musan_music_item
    speech_csv = os.path.join(csv_folder, "musan_speech.csv")
    csv_dct[speech_csv] = musan_speech_item

    # Prepare csv if necessary
    for csv_file, items in csv_dct.items():

        if not os.path.isfile(csv_file):
            if csv_file in [noise_csv, bg_music_csv, pointsrc_noises_csv]:
                prepare_aug_csv(items, csv_file, overlap, max_noise_len)
            else:
                prepare_aug_csv(items, csv_file, overlap=None, max_length=None)
    # ---------------------------------------------------------------------------------------
    # concate csv
    combine_music_noise_csv = os.path.join(csv_folder, "combine_music_noise.csv")
    combine_sim_small_medium_rev_csv = os.path.join(
        csv_folder, "combine_sim_small_medium_rev.csv"
    )
    combine_sim_rev_csv = os.path.join(csv_folder, "combine_sim_rev.csv")
    concat_csv(combine_music_noise_csv, bg_music_csv, noise_csv)

    concat_csv(combine_sim_small_medium_rev_csv, sim_small_csv, sim_medium_csv)
    concat_csv(combine_sim_rev_csv, sim_small_csv, sim_medium_csv, sim_large_csv)

    logger.info(
        f"Prepare the speech augment dataset Done, csv files is in {csv_folder}, "
        f"wavs in {save_folder}."
    )


def prepare_aug_csv(items, csv_file, overlap, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.

    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """

    with open(csv_file, "w") as w:
        if max_length is None:
            w.write("ID,duration,wav,sr,tot_frame,wav_format\n")
        else:
            w.write("ID,duration,wav,sr,tot_frame,start,stop,wav_format\n")
        # copy ordinary wav.
        for item in track(items):
            if not os.path.isdir(os.path.dirname(item[1])):
                os.makedirs(os.path.dirname(item[1]))
            shutil.copyfile(item[0], item[1])
            filename = item[1]
            # Read file for duration/channel info
            signal, rate = torchaudio.load(filename)

            # Ensure only one channel
            if signal.shape[0] > 1:
                signal = signal[0].unsqueeze(0)
                torchaudio.save(filename, signal, rate)

            ID, ext = os.path.basename(filename).split(".")
            duration = signal.shape[1] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)

                overlap_size = int(overlap * max_length * rate)
                max_size = int(max_length * rate)
                tot_size = int(duration * rate)
                offset = 0
                counter = 0
                while offset + max_size <= tot_size:
                    start = offset
                    stop = start + max_size
                    offset += int(max_size - overlap_size)

                    new_filename = filename[: -len(f".{ext}")] + f"_{counter}.{ext}"
                    torchaudio.save(new_filename, signal[:, start:stop], rate)
                    csv_row = (
                        f"{ID}_{counter}",
                        str((stop - start) / rate),
                        new_filename,
                        str(rate),
                        str(stop - start),
                        str(start),
                        str(stop),
                        ext,
                    )
                    w.write(",".join(csv_row) + "\n")
                    counter += 1

                if offset + (max_size / 2) <= tot_size:
                    start = tot_size - max_size
                    stop = tot_size

                    new_filename = filename[: -len(f".{ext}")] + f"_{counter}.{ext}"
                    torchaudio.save(new_filename, signal[:, start:stop], rate)
                    csv_row = (
                        f"{ID}_{counter}",
                        str((stop - start) / rate),
                        new_filename,
                        str(rate),
                        str(stop - start),
                        str(start),
                        str(stop),
                        ext,
                    )
                    w.write(",".join(csv_row) + "\n")
            elif max_length is not None and duration <= max_length:
                w.write(
                    ",".join(
                        (
                            ID,
                            str(duration),
                            filename,
                            str(rate),
                            str(signal.shape[1]),
                            str(0),
                            str(signal.shape[1]),
                            ext,
                        )
                    )
                    + "\n"
                )
            else:
                w.write(
                    ",".join(
                        (
                            ID,
                            str(duration),
                            filename,
                            str(rate),
                            str(signal.shape[1]),
                            ext,
                        )
                    )
                    + "\n"
                )


def concat_csv(out_file, *csv_files):
    pd_list = []
    for f in csv_files:
        pd_list.append(pd.read_csv(f, sep=",", header=0))
    out = pd.concat(pd_list)
    out.to_csv(out_file, sep=",", header=True, index=False)


if __name__ == "__main__":

    # Start
    parser = argparse.ArgumentParser(
        description=""" Prepare speech augmention csv files.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )

    # Options
    parser.add_argument(
        "--openrir-folder",
        type=str,
        default="/tsdata/ASR",
        help="where has openslr rir.",
    )

    parser.add_argument(
        "--musan-folder",
        type=str,
        default="/tsdata/ASR",
        help="where has openslr musan.",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="/work1/ldx/speech_aug_2_new",
        help="Save noise clips for online speechaug, suggest in SSD.",
    )
    parser.add_argument(
        "--force-clear",
        type=str,
        action=StrToBoolAction,
        default=True,
        choices=["true", "false"],
        help="force clear",
    )
    parser.add_argument(
        "--max-noise-len",
        type=float,
        default=2.015,
        help="the maximum noise length in seconds. Noises longer than this will be cut into pieces",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0,
        help="the proportion of overlapping for every chunk",
    )
    parser.add_argument("csv_aug_folder", type=str, help="csv file folder.")

    # End
    logger.info(" ".join(sys.argv))
    args = parser.parse_args()
    # assert args.max_noise_len > 0.4
    if args.max_noise_len == 0:
        args.max_noise_len = None

    prepare_speech_aug(
        args.openrir_folder,
        args.musan_folder,
        csv_folder=args.csv_aug_folder,
        save_folder=args.save_folder,
        max_noise_len=args.max_noise_len,
        overlap=args.overlap,
        force_clear=args.force_clear,
    )
