# CDMA

## 1. Dependencies
```bash
conda env create -f environment.yml
```

## 2. Install ASV-subtools & Kaldi
Please refer to the installation details in https://github.com/matln/asv-subtools#ready-to-start.

Change to your own Kaldi & ASV-subtools path in `path.sh`.

```bash
export KALDI_ROOT=<your path>
export SUBTOOLS=<your path>
```

## 3. Prepare the training & eval data

It is recommended to change the `stage` and `endstage` to run step by step.

```bash
bash preprocess.sh
```

## 4. Run the model

1. Pretraining
  ```bash
  bash run_resnet34_pretrain.sh
  ```
2. Supervised domain adaptation
  ```bash
  bash run_resnet34_SDA.sh
  ```
3. Unsupervised domain adaptation
  ```bash
  bash run_resnet34_UDA.sh
  ```
4. Optimize the MMD loss in the embedding space
  ```bash
  bash run_resnet34_emb_mmd.sh
  ```
5. Finetuning
  ```bash
  bash run_resnet34_FT.sh
  ```
  
