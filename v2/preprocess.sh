#!/bin/bash

### Start 
. ./path.sh

# --------------------------------------------------------------------------------------------- #

# ==> Make sure the audio datasets (voxceleb1, voxceleb2, RIRS and Musan) have been downloaded by yourself.
voxceleb1_path=/home/lijianchen/pdata/VoxCeleb1_v2
voxceleb2_path=/home/lijianchen/pdata/VoxCeleb2
voices_path=/home/lijianchen/pdata/VOiCES_Box_unzip

stage=0
endstage=0

# --------------------------------- dataset preparation ------------------------------- #
if [[ $stage -le 0 && 0 -le $endstage ]]; then
	# Prepare the data/voxceleb1_train, data/voxceleb1_test and data/voxceleb2_train.
    local/make_voxceleb1_v2.pl $voxceleb1_path dev data/voxceleb1_dev
    local/make_voxceleb1_v2.pl $voxceleb1_path test data/voxceleb1_test
	local/make_voxceleb2.pl $voxceleb2_path dev data/voxceleb2_dev
    python local/make_voices.py --root-dir $voices_path --dataset dev
    python local/make_voices.py --root-dir $voices_path --dataset dev_label
    python local/make_voices.py --root-dir $voices_path --dataset eval

    combine_data.sh data/voxceleb1o2_dev data/voxceleb1_dev data/voxceleb2_dev

    python3 tools/make_utt2dur.py --data-dir=data/voices_eval
    
    python3 local/split_dev.py

fi

# ------------------------------ preprocess to generate chunks ------------------------- #
seed=1024

if [[ $stage -le 1 && 1 -le $endstage ]]; then
    ${speakernet}/pipelines/prepare_train_csv.sh \
        --seed ${seed} \
        --duration 4.0 \
        --min_duration 3.8 \
        --drop_last_duration 2.0 \
        --overlap 0 \
        --valid_split_from_trainset "false" \
        --amp_th 0 \
        data/voxceleb2_dev exp/egs/vox2_4s

    ${speakernet}/pipelines/prepare_train_csv.sh \
        --seed ${seed} \
        --duration 4.0 \
        --min_duration 2.0 \
        --drop_last_duration 2.0 \
        --overlap 0.1 \
        --valid_split_from_trainset "false" \
        --amp_th 0 \
        data/voices_train exp/egs/voices_train
fi

# ------------------------------ Prepare augmentation csv file ------------------------- #
csv_folder=exp/aug_csv

if [[ $stage -le 2 && 2 -le $endstage ]]; then
    python3 ${speakernet}/pipelines/prepare_aug_csv.py \
        --openrir-folder=/data/corpus/rirs_noises \
        --musan-folder=/data/corpus/MUSAN \
        --save-folder=/home/lijianchen/pdata/noise_4.015s \
        --max-noise-len=4.015 \
        --overlap=0 \
        --force-clear=true \
        ${csv_folder}
fi
