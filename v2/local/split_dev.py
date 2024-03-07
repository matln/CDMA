import os
import random

random.seed(1024)


dev_dir = "data/voices_dev_label"
valid_dir = "data/voices_val"
train_dir = "data/voices_train"
train_dir1 = "data/voices_train_unlabeled"

os.makedirs(valid_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(train_dir1, exist_ok=True)

spk2utts = {}
with open(f"{dev_dir}/spk2utt", "r") as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip().split()
        spk = line[0]
        utts = line[1:]
        spk2utts[spk] = utts

utt2wav = {}
with open(f"{dev_dir}/wav.scp", "r") as fr:
    lines = fr.readlines()
    for line in lines:
        utt, wav = line.strip().split()
        utt2wav[utt] = wav

spks = list(spk2utts.keys())
valid_spks = random.sample(spks, 40)
train_spks = list(set(spks) - set(valid_spks))
train_spks.sort()

valid_utts = []
with open(f"{valid_dir}/spk2utt", "w") as fw:
    for spk in valid_spks:
        utts = spk2utts[spk]
        valid_utts.extend(utts)
        fw.write(f"{spk} {' '.join(utts)}\n")
with open(f"{valid_dir}/wav.scp", "w") as fw:
    for utt in valid_utts:
        fw.write(f"{utt} {utt2wav[utt]}\n")

if os.system(f"spk2utt_to_utt2spk.pl data/voices_val/spk2utt >data/voices_val/utt2spk") != 0:
    print(f"Error creating utt2spk file in directory voices_val")
os.system(f"env LC_COLLATE=C fix_data_dir.sh data/voices_val")
if os.system(f"env LC_COLLATE=C validate_data_dir.sh --no-text --no-feats data/voices_val") != 0:
    print(f"Error validating directory data/voices_val")

# valid trials
os.makedirs(f"{valid_dir}/trials", exist_ok=True)
# with open(f"{valid_dir}/trials/val_trials", "w") as fw:
#     for i, utt1 in enumerate(valid_utts):
#         spk1 = utt1.split('-')[0]
#         for utt2 in valid_utts[i + 1 :]:
#             spk2 = utt2.split('-')[0]
#             if spk1 == spk2:
#                 fw.write(f"{utt1} {utt2} target\n")
#             else:
#                 fw.write(f"{utt1} {utt2} nontarget\n")

with open(f"{valid_dir}/trials/val_trials1", "w") as fw:
    with open(f"data/voices_dev_label/trials/dev_label_trials", "r") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split()
            utt1 = line[0]
            utt2 = line[1]
            if utt1 in valid_utts and utt2 in valid_utts:
                fw.write(f"{utt1} {utt2} {line[2]}\n")











train_utts = []
with open(f"{train_dir}/spk2utt", "w") as fw:
    for spk in train_spks:
        utts = spk2utts[spk]
        train_utts.extend(utts)
        fw.write(f"{spk} {' '.join(utts)}\n")
with open(f"{train_dir}/wav.scp", "w") as fw:
    for utt in train_utts:
        fw.write(f"{utt} {utt2wav[utt]}\n")

if os.system(f"spk2utt_to_utt2spk.pl data/voices_train/spk2utt >data/voices_train/utt2spk") != 0:
    print(f"Error creating utt2spk file in directory voices_train")
os.system(f"env LC_COLLATE=C fix_data_dir.sh data/voices_train")
if os.system(f"env LC_COLLATE=C validate_data_dir.sh --no-text --no-feats data/voices_train") != 0:
    print(f"Error validating directory data/voices_train")


with open(f"{train_dir1}/utt2spk", "w") as fw1, open(f"{train_dir1}/wav.scp", "w") as fw2, open(f"{train_dir1}/spk2utt", "w") as fw3:
    for utt in train_utts:
        fw1.write(f"{utt.split('-', 1)[1]} {utt.split('-', 1)[1]}\n")
        fw2.write(f"{utt.split('-', 1)[1]} {utt2wav[utt]}\n")
        fw3.write(f"{utt.split('-', 1)[1]} {utt.split('-', 1)[1]}\n")

os.system(f"env LC_COLLATE=C fix_data_dir.sh data/voices_train_unlabeled")
if os.system(f"env LC_COLLATE=C validate_data_dir.sh --no-text --no-feats data/voices_train_unlabeled") != 0:
    print(f"Error validating directory data/voices_train_unlabeled")
