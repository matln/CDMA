#!/bin/bash

### Start 
. ./path.sh

# --------------------------------------------------------------------------------------------- #
seed=1024
feature="fbank_64"

source_egs_dir=exp/egs/${feature}-voxceleb1o2_train-400-sequential-ssd
target_egs_dir=exp/egs/${feature}-voices_dev-400-sequential-novad

stage=0
endstage=0

# --------------------------------- training (pytorch) ------------------------------------ #
model_dir="UDA_resnet34_${feature}"

if [[ $stage -le 0 && 0 -le $endstage ]]; then
  time="$(date "+%Y-%m-%d_%H:%M:%S")"
  echo "$time" | tee timestamp

  "${SUBTOOLS}"/runPytorchLauncher.sh --endstage 0 --master_addr "127.0.0.1" --pdb "false" \
    run_resnet34.py \
      --debug="false" \
      --model-dir=${model_dir} \
      --stage=0 \
      --gpu-id=0 \
      --seed=$seed \
      --source-egs-dir=${source_egs_dir} \
      --target-egs-dir=${target_egs_dir} \
      --train-time-string="$time" \
      --feature=$feature \
      --use-step="false" \
      --num-chunks=4 \
      --batch-size=128 \
      --learn-rate=0.001 \
      --lr-scheduler="MultiStepLR" \
      --milestones="10,20" \
      --epochs=30 \
      --exist-model="exp/resnet34_fbank_64/2021-07-18_01:13:37/params/9.7744.params" \
      --mode="UDA"
fi

# ------------------------- extract the embeddings of voices-dev ----------------------------- #
# Extract embeddings on another computer with 8 GPUs (to run in parallel with the training stage)
[[ "$?" != "0" ]] && exit 1

time=
: =${time:=$(cat timestamp)}

if [[ $stage -le 1 && 1 -le $endstage ]]; then
  "${SUBTOOLS}"/runPytorchLauncher.sh --endstage 1 --master_addr "127.0.0.1" --pdb "false" \
    run_resnet34.py \
      --model-dir=${model_dir} \
      --stage=1 \
      --gpu-id=0,1,2,3,4,5,6,7 \
      --seed=$seed \
      --extract-positions="near" \
      --extract-data="voices_dev_cmn,voxceleb1_train_cmn" \
      --extract-epochs="$(seq -s "," 30)" \
      --train-time-string="$time" \
      --feature=$feature
fi

# --------------------------------- Back-end scoring ---------------------------------- #
[[ "$?" != "0" ]] && exit 1

if [[ $stage -le 2 && 2 -le $endstage ]]; then
  vectordir="exp/${model_dir}/${time}"
  [ ! -d $vectordir ] && echo "$vectordir doesn't exist" && exit 1

  epochs=$(cat exp/${model_dir}/${time}/config/extracted_epochs)
  echo "$epochs"

  results_tmp=$vectordir/results/tmp.results
  : > $results_tmp

  local/gather_results.sh --prefix $feature --score plda --lda true --submean true --vectordir $vectordir --evalset "voices_dev_cmn" \
    --metric "minDCF" --task "voices_dev" --task_name "voices_dev2" --epochs "$epochs" --positions "near" \
    --trainset "voxceleb1_train_cmn" --submean_trainset "voices_dev_cmn" --force true

  cat $results_tmp
fi

# ------------------------- extract the embeddings of voices-eval ----------------------------- #
[[ "$?" != "0" ]] && exit 1

# select the model with the best performance on voices_dev
epochs="13"

if [[ $stage -le 3 && 3 -le $endstage ]]; then
  "${SUBTOOLS}"/runPytorchLauncher.sh --endstage 1 --master_addr "127.0.0.1" --pdb "false" \
    run_resnet34.py \
      --model-dir=${model_dir} \
      --stage=1 \
      --gpu-id=0,1,2,3,4,5,6,7 \
      --seed=$seed \
      --extract-positions="near" \
      --extract-data="voices_eval_cmn" \
      --extract-epochs="${epochs}" \
      --train-time-string="$time" \
      --feature=$feature
fi

# --------------------------------- Back-end scoring ---------------------------------- #
[[ "$?" != "0" ]] && exit 1

if [[ $stage -le 4 && 4 -le $endstage ]]; then
  vectordir="exp/${model_dir}/${time}"
  [ ! -d $vectordir ] && echo "$vectordir doesn't exist" && exit 1

  results_tmp=$vectordir/results/tmp.results
  : > $results_tmp

  local/gather_results.sh --prefix $feature --score plda --lda true --submean true --vectordir $vectordir \
    --evalset "voices_eval_cmn" --metric "minDCF" --task "voices_eval" --task_name "voices_eval2" --epochs "$epochs" \
    --positions "near" --trainset "voxceleb1_train_cmn" --submean_trainset "voices_dev_cmn" --force true

  cat $results_tmp
fi

### All Done ###
