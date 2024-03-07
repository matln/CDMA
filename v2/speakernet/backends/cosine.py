"""
Copyright 2022 Zhengyang Chen
          2022 Jianchen Li
"""

import os
import torch
import logging
import kaldiio
import numpy as np
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_mean(scp_path):
    emb_num = 0
    mean_emb = None

    for _, emb in kaldiio.load_scp_sequential(scp_path):
        # emb = emb / np.sqrt(np.sum(emb ** 2))
        if mean_emb is None:
            mean_emb = np.zeros_like(emb)
        mean_emb += emb
        emb_num += 1

    return mean_emb / emb_num


def score_trials(eval_embs_path, score_path, mean_emb, trials, quality_measures):
    # Each embedding may be accessed multiple times, here we pre-load them
    # into the memory
    utt2emb = {}
    if "embedding" in quality_measures:
        # Magnitude of embeddings, generated for score calibration
        utt2emb_mgt = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_embs_path):
        # emb = emb / np.sqrt(np.sum(emb ** 2))
        emb = emb - mean_emb
        utt2emb[utt] = emb
        if "embedding" in quality_measures:
            utt2emb_mgt[utt] = np.sqrt(np.sum(emb ** 2))

    if "embedding" in quality_measures:
        output_mgt_path = f"{os.path.dirname(score_path)}/emb_mgt.qual"
        f_mgt = open(output_mgt_path, "w")

    with open(trials, "r") as r, open(score_path, "w") as w:
        lines = r.readlines()
        enrolls = []
        tests = []
        for line in lines:
            segs = line.strip().split()
            emb1, emb2 = utt2emb[segs[0]], utt2emb[segs[1]]
            enrolls.append(torch.from_numpy(emb1))
            tests.append(torch.from_numpy(emb2))
            if "embedding" in quality_measures:
                if utt2emb_mgt[segs[0]] < utt2emb_mgt[segs[1]]:
                    # enroll test low_quality high_quality
                    f_mgt.write(f"{segs[0]} {segs[1]} {utt2emb_mgt[segs[0]]} {utt2emb_mgt[segs[1]]}\n")
                else:
                    f_mgt.write(f"{segs[0]} {segs[1]} {utt2emb_mgt[segs[1]]} {utt2emb_mgt[segs[0]]}\n")
        enrolls = torch.stack(enrolls)
        tests = torch.stack(tests)
        cos_score = F.cosine_similarity(enrolls, tests, dim=1)

        for i, line in enumerate(lines):
            segs = line.strip().split()
            w.write(f"{segs[0]} {segs[1]} {cos_score[i].item():.5f}\n")
    if "embedding" in quality_measures:
        f_mgt.close()


def compute_cosine_score(
    emb_dir, evalset, trials, submean, submean_set=None, force=False, quality_measures=[]
):
    """
    force : bool
        Force score the trials
    """
    if submean:
        mean_emb_path = os.path.join(emb_dir, submean_set, "mean_emb.npy")
        if not os.path.exists(mean_emb_path):
            scp_path = os.path.join(emb_dir, submean_set, "xvector.scp")
            logger.info("Computing the mean statistics from {}.".format(scp_path))
            mean_emb = compute_mean(scp_path)
            np.save(mean_emb_path, mean_emb)
        else:
            mean_emb = np.load(mean_emb_path)
    else:
        mean_emb = 0.0

    # scoring trials
    eval_embs_path = os.path.join(emb_dir, evalset, "xvector.scp")
    scores_dir = os.path.join(emb_dir, evalset, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    suffix = ".score" if not submean else "_submean.score"
    score_path = os.path.join(scores_dir, "cosine_" + os.path.basename(trials) + suffix)
    output_mgt_path = f"{scores_dir}/emb_mgt.qual"
    if force or not os.path.exists(score_path):
        logger.info("Computing scores ...")
        score_trials(eval_embs_path, score_path, mean_emb, trials, quality_measures)

    return score_path, output_mgt_path


if __name__ == "__main__":
    pass
