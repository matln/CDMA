#!/bin/bash

### Start
. ./path.sh

model="CDMA"

# time=$(cat tmp/train1.timestamp)
time=2024-02-07_19:05:55
extracted_epochs=""

# trials="val_trials1"
# evalset="voices_val"
trials="eval_trials"
evalset="voices_eval"

# export speakernet=exp/${model}/${time}/backup/speakernet

exp_dir="exp/${model}/${time}" && [ ! -d $exp_dir ] && echo "$exp_dir doesn't exist" && exit 1

python3 ${speakernet}/pipelines/score.py \
    --exp-dir=$exp_dir \
    --epochs="$extracted_epochs" \
    --trials="$trials" \
    --evalset="${evalset}" \
    --submean="false" \
    --submean-set="voices_train_pseudo" \
    --score-norm-method="" \
    --cohort-set="voices_train_pseudo" \
    --top-n=300 \
    --average-cohort="false" \
    --score-calibration="false" \
    --cali-dev-set="voices_dev_label" \
    --cali-dev-trials="dev_trials" \
    --quality-measures="duration imposter" \
    --return-thresh="false" \
    --ptarget=0.01 \
    --force="true"
