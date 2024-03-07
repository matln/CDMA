#!/bin/bash

### Start
. ./path.sh

model="CDMA"
hparam_file="CDMA.yaml"
resume_training="false"
debug="false"

# Specify the time to resume training
time=

[[ $time != "" ]] && resume_training="true" && export speakernet=exp/${model}/${time}/backup/speakernet
: =${time:=$(date "+%Y-%m-%d_%H:%M:%S")}
mkdir -p tmp
# echo $time | tee tmp/train.timestamp
echo $time > tmp/train2.timestamp

${speakernet}/pipelines/launcher.sh --master_addr "127.0.0.1" --pdb "false" \
    $(dirname "$speakernet")/train.py --hparams-file hparams/${hparam_file} \
        --model-dir="$model" \
        --gpu-id=0 \
        --debug="$debug" \
        --mixed-prec="true" \
        --train-time-string="$time" \
        --resume-training=$resume_training
