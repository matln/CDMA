#!/bin/bash

### Start
. ./path.sh

model="pretrain"
# model="CDMA"

# time=$(cat tmp/train1.timestamp)
time=2022-11-19_17:34:35
xvector_extracted_epochs=""

# export speakernet=exp/${model}/${time}/backup/speakernet
#
python3 ${speakernet}/pipelines/extract_embeddings.py \
	--model-dir="$model" \
	--gpu-id="0 1" \
	--train-time-string="$time" \
	--extract-epochs="$xvector_extracted_epochs" \
    --extract_data="voxceleb2_dev_subset" \
    --jobs-per-gpu=2 \
    --lower-epoch=-1

# python3 ${speakernet}/pipelines/extract_embeddings.py \
# 	--model-dir="$model" \
# 	--gpu-id="2 3 4 5" \
# 	--train-time-string="$time" \
# 	--extract-epochs="$xvector_extracted_epochs" \
#     --extract_data="voices_val" \
#     --jobs-per-gpu=2 \
#     --lower-epoch=0

# python3 ${speakernet}/pipelines/extract_embeddings.py \
# 	--model-dir="$model" \
# 	--gpu-id="1" \
# 	--train-time-string="$time" \
# 	--extract-epochs="$xvector_extracted_epochs" \
#     --extract_data="voices_eval" \
#     --jobs-per-gpu=2 \
#     --lower-epoch=0
