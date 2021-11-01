# -*- coding:utf-8 -*-

import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("--root-dir", type=str, default="/data/corpus/VOiCES/VOiCES_Box_unzip", help="")
parser.add_argument("--dataset", type=str, default="", choices=["dev", "dev_label", "eval"], help="")
parser.add_argument("--out-dir", type=str, default="./data", help="")

args = parser.parse_args()

data_dir = "{}/voices_{}".format(args.out_dir, args.dataset)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

f1 = open("{}/wav.scp".format(data_dir), "w")
f2 = open("{}/utt2spk".format(data_dir), "w")
f3 = open("{}/trials".format(data_dir), "w")
utt2spk = {}
if args.dataset == "dev":
    wav_dir = os.path.join(args.root_dir, "Development_Data/Speaker_Recognition/sid_dev") 
    for spk in os.listdir(wav_dir):
        if spk[:1] == ".": continue
        for wav in os.listdir(os.path.join(wav_dir, spk)):
            utt = wav[:-4]
            full_path = os.path.join(wav_dir, spk, wav)
            utt2spk[utt] = utt
            f1.write("{} {}\n".format(utt, full_path))
            f2.write("{} {}\n".format(utt, utt))

    trials_path = "{}/Development_Data/Speaker_Recognition/"\
            "sid_dev_lists_and_keys/dev-trial-keys.lst".format(args.root_dir)
    with open(trials_path, "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            enroll = line[0]
            test = os.path.basename(line[1])[:-4]
            if line[2] == "imp":
                target = "nontarget"
            elif line[2] == "tgt":
                target = "target"
            f3.write("{} {} {}\n".format(enroll, test, target))
# 开发集有说话人标签
elif args.dataset == "dev_label":
    wav_dir = os.path.join(args.root_dir, "Development_Data/Speaker_Recognition/sid_dev") 
    for spk in os.listdir(wav_dir):
        if spk[:1] == ".": continue
        for wav in os.listdir(os.path.join(wav_dir, spk)):
            utt = "{}-{}".format(spk, wav[:-4])
            full_path = os.path.join(wav_dir, spk, wav)
            utt2spk[wav[:-4]] = spk
            f1.write("{} {}\n".format(utt, full_path))
            f2.write("{} {}\n".format(utt, spk))

    trials_path = "{}/Development_Data/Speaker_Recognition/"\
            "sid_dev_lists_and_keys/dev-trial-keys.lst".format(args.root_dir)
    with open(trials_path, "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            # assert utt2spk[line[0]] == line[0].split('-')[5]
            enroll = "{}-{}".format(utt2spk[line[0]], line[0])
            test_name = os.path.basename(line[1])[:-4]
            # assert utt2spk[test_name] == test_name.split('-')[5]
            test = "{}-{}".format(utt2spk[test_name], test_name)
            if line[2] == "imp":
                target = "nontarget"
            elif line[2] == "tgt":
                target = "target"
            f3.write("{} {} {}\n".format(enroll, test, target))
else:
    wav_dir = os.path.join(args.root_dir, "Speaker_Recognition/sid_eval")
    for wav in os.listdir(wav_dir):
        full_path = os.path.join(wav_dir, wav)
        f1.write("{} {}\n".format(wav[:-4], full_path))
        f2.write("{} {}\n".format(wav[:-4], wav[:-4]))

    trials_path = "{}/VOiCES_challenge_2019_post-eval-release/"\
            "VOiCES_challenge_2019_eval.SID.trial-keys.lst".format(args.root_dir)
    with open(trials_path, "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            enroll = line[0]
            test = line[1][:-4]
            if line[2] == "imp":
                target = "nontarget"
            elif line[2] == "tgt":
                target = "target"
            f3.write("{} {} {}\n".format(enroll, test, target))
f1.close()
f2.close()
f3.close()

if os.system("utt2spk_to_spk2utt.pl {0}/utt2spk >{0}/spk2utt".format(data_dir)) != 0:
    print("Error creating spk2utt file in directory data/voices_{}".format(args.dataset))
os.system("env LC_COLLATE=C fix_data_dir.sh {}".format(data_dir))
if os.system("env LC_COLLATE=C validate_data_dir.sh --no-text --no-feats {}".format(data_dir)) != 0:
    print("Error validating directory data/voices_{}".format(args.dataset))
