#!/bin/bash
# Copyright 2022 Jianchen Li

set -e

stage=1
endstage=1

suffix=
limit_dur=60

# Get chunk egs
duration=2.0
min_duration=2.0
sample_type="sequential"  # sequential | speaker_balance | random_segment
chunk_num=-1
scale=1.5
overlap=0.1
frame_overlap=0.015
drop_last_duration=0.2
valid_split_type="--total-spk"
valid_num_spks=1024
valid_chunk_num=2
valid_sample_type="sequential" # With sampleSplit type [--total-spk] and sample type [every_utt], we will get enough spkers as more
                              # as possible and finally we get valid_num_utts * valid_chunk_num = 1024 * 2 = 2048 valid chunks.
valid_split_from_trainset="true"
seed=1024
amp_th=0.0005      # 50 / (1 << 15)
samplerate=16000

expected_files="utt2spk,spk2utt,wav.scp"

. parse_options.sh

if [[ $# != 2 && $# != 3 ]]; then
  echo "[exit] Num of parameters is not equal to 2 or 3"
  echo "usage:$0 <data-dir> <egs-dir>"
  exit 1
fi

# Key params
train_data=$1
egsdir=$2
valid_data=$3

if [[ $stage -le 0 && 0 -le $endstage ]]; then
    echo "$0: stage 0"

    if [[ $suffix != "" ]]; then
        \rm -rf ${train_data}${suffix}
        cp -r ${train_data} ${train_data}${suffix}
        train_data=${train_data}${suffix}
    fi

    # Remove speakers with too few training samples
    ${speakernet}/pipelines/modules/remove_short_utt.sh --limit-dur $limit_dur \
        ${train_data} $(echo "$min_duration + $frame_overlap" | bc) || exit 1
fi

if [[ $stage -le 1 && 1 -le $endstage ]]; then
    echo "$0: stage 1"
    [ "$egsdir" == "" ] && echo "The egsdir is not specified." && exit 1

	# valid: validation
	python3 ${speakernet}/pipelines/modules/get_chunk_csv.py \
		--duration=$duration \
		--min-duration=$min_duration \
		--sample-type=$sample_type \
		--chunk-num=$chunk_num \
		--scale=$scale \
		--overlap=$overlap \
		--frame-overlap=$frame_overlap \
		--drop-last-duration=$drop_last_duration \
		--valid-split-type=$valid_split_type \
		--valid-num-spks=$valid_num_spks \
		--valid-chunk-num=$valid_chunk_num \
		--valid-sample-type=$valid_sample_type \
        --valid-dir="${valid_data}" \
        --valid-split-from-trainset=$valid_split_from_trainset \
		--seed=$seed \
		--amp-th=$amp_th \
		--samplerate=$samplerate \
        --expected-files="$expected_files" \
		${train_data} ${egsdir} || exit 1
fi

exit 0
