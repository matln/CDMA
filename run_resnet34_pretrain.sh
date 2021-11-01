#!/bin/bash

### Start 
. ./path.sh

# --------------------------------------------------------------------------------------------- #
seed=1024
feature="fbank_64"

source_egs_dir=exp/egs/${feature}-voxceleb1o2_train-400-sequential-ssd

stage=0
endstage=0

# ------------------------ training (pytorch) & extract xvector ------------------------- #
model_dir="resnet34_${feature}"
time=
: =${time:=$(date "+%Y-%m-%d_%H:%M:%S")}

if [[ $stage -le 0 && 0 -le $endstage ]]; then
  echo $time

  xvector_stage=0  # 0 or 1
  xvector_endstage=0

  "${SUBTOOLS}"/runPytorchLauncher.sh --endstage ${xvector_endstage} --pdb "false" \
    run_resnet34.py \
      --debug="false" \
      --model-dir=${model_dir} \
      --stage=$xvector_stage \
      --gpu-id=0 \
      --seed=$seed \
      --source-egs-dir=${source_egs_dir} \
      --extract-positions="near" \
      --extract-data="voices_dev_label_cmn,voices_eval_cmn,voxceleb1_train_cmn" \
      --extract-epochs="" \
      --train-time-string="$time" \
      --use-fast-loader="true" \
      --feature=$feature \
      --margin-loss="true" \
      --use-step="true" \
      --batch-size=256 \
      --learn-rate=0.02 \
      --lr-scheduler="reduceP" \
      --epochs=12 \
      --mixed-prec="false" \
      --mode="pretrain"
fi

# --------------------------------- Back-end scoring ---------------------------------- #
[[ "$?" != "0" ]] && exit 1

if [[ $stage -le 1 && 1 -le $endstage ]]; then
  metric="minDCF" # eer | minDCF #
  task="voices_eval"
  task_name="voices_eval1"
  evalset="voices_eval_cmn"
  trainset="voices_dev_label_cmn"
  # trainset="voxceleb1_train_cmn"
  submean_trainset="voices_dev_label_cmn"

  # epochs=$(cat exp/${model_dir}/${time}/config/extracted_epochs)
  epochs="9.7744"
  echo "$epochs"

  vectordir="exp/${model_dir}/${time}"
  [ ! -d $vectordir ] && echo "$vectordir doesn't exist" && exit 1

  results_tmp=$vectordir/results/tmp.results
  : > $results_tmp

  local/gather_results.sh --prefix $feature --score "plda" --lda true --submean true --vectordir $vectordir --evalset $evalset --metric $metric \
    --task $task --task_name $task_name --epochs "$epochs" --positions "near" --trainset $trainset --submean_trainset $submean_trainset --force true
fi
### All Done ###
