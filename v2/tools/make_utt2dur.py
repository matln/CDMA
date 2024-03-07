"""
Copyright 2022 Jianchen Li
"""
import os
import sys
import argparse
import torchaudio

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

from speakernet.utils.utils import read_scp
from speakernet.utils.rich_utils import track


parser = argparse.ArgumentParser(description="")
parser.add_argument("--data-dir", type=str, default="",
                    help="The directory that should contain wav.scp utt2spk spk2utt")
args = parser.parse_args()

wav_scp = read_scp(f"{args.data_dir}/wav.scp")

with open(f"{args.data_dir}/utt2dur", "w") as fw:
    for utt, wav in track(wav_scp.items()):
        sig, lr = torchaudio.load(wav)
        dur = sig.size(1) / lr
        fw.write(f"{utt} {dur}\n")
