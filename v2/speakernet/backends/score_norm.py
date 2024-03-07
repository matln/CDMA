"""
Copyright 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
          2022 Jianchen Li
"""

import os
import torch
import kaldiio
import logging
import numpy as np
import torch.nn.functional as F

from speakernet.utils.utils import read_scp, read_trials


logger = logging.getLogger(__name__)


def get_mean_std(embs, cohorts, top_n, cosine=True):
    # Cosine similarity or inner product
    embs = torch.from_numpy(embs)
    cohorts = torch.from_numpy(cohorts)
    if cosine:
        embs = F.normalize(embs, p=2, dim=1)
        cohorts = F.normalize(cohorts, p=2, dim=1)
    emb_cohort_scores = torch.mm(embs, cohorts.T)
    # np.sort is too slow (especially for vox1-H and vox-E), and np.partition (or np.argpartition) has unstable performance.
    emb_cohort_scores_topn, _ = torch.topk(emb_cohort_scores, top_n, dim=1, sorted=False)

    emb_means = torch.mean(emb_cohort_scores_topn, dim=1)
    emb_stds = torch.std(emb_cohort_scores_topn, dim=1, unbiased=False)

    return emb_means.numpy(), emb_stds.numpy()


def split_embedding(utts, emb_scp, mean_emb):
    embs = []
    utt2idx = {}
    utt2emb = {}
    for utt, emb in kaldiio.load_scp_sequential(emb_scp):
        # emb = emb / np.sqrt(np.sum(emb ** 2))
        emb = emb - mean_emb
        utt2emb[utt] = emb

    for i, utt in enumerate(utts):
        embs.append(utt2emb[utt])
        utt2idx[utt] = i

    return np.array(embs), utt2idx


def normalize(
    score_norm_method,
    top_n,
    score_file,
    cohort_embs_scp,
    eval_embs_scp,
    mean_emb_path,
    quality_measures,
):
    """
    score_norm_method : str
        Can only be asnorm or snorm
    top_n : int
        Selecting only the top_n speakers with largest scores from the normalization cohort.
    score_file : str
        score file that needs to be normalized.
    cohort_embs_scp : str
        The embedding scp file for cohort set.
    eval_embs_scp : str
        Pre-extracted evalset embedding file.
    mean_emb_path : str
        Pre-computed mean statistic
    quality_measures : list
        Various quality measurements. If contains "imposter", then output mean imposter score.

    Recommend: Analysis of Score Normalization in Multilingual Speaker Recognition. Interspeech 2017.
    """
    if score_norm_method == "snorm":
        score_norm_file = score_file.replace(".score", "_snorm.score")
    else:
        score_norm_file = score_file.replace(".score", f"_asnorm_top{top_n}.score")

    if not os.path.exists(mean_emb_path):
        logger.info("Do not subtract mean statistic when normalizing scores")
        mean_emb = 0.0
    else:
        if not os.path.exists(mean_emb_path):
            logger.error("Mean_emb file dose not exist!")
        mean_emb = np.load(mean_emb_path)

    # get embedding
    enrolls, tests, _ = zip(*read_trials(score_file))
    # Remove the overlap and sort
    enrolls = sorted(list(set(enrolls)))
    tests = sorted(list(set(tests)))
    enroll_embs, enroll_utt2idx = split_embedding(enrolls, eval_embs_scp, mean_emb)
    test_embs, test_utt2idx = split_embedding(tests, eval_embs_scp, mean_emb)

    cohorts = list(read_scp(cohort_embs_scp).keys())
    cohort_embs, _ = split_embedding(cohorts, cohort_embs_scp, mean_emb)

    if score_norm_method == "asnorm":
        top_n = top_n
    elif score_norm_method == "snorm":
        top_n = cohort_embs.shape[0]
    else:
        raise ValueError(score_norm_method)
    logger.info("Computing mean and std of cohort scores ...")
    enroll_means, enroll_stds = get_mean_std(enroll_embs, cohort_embs, top_n)
    test_means, test_stds = get_mean_std(test_embs, cohort_embs, top_n)

    mean_imposter_path = f"{os.path.dirname(score_file)}/mean_imposter.qual"
    if "imposter" in quality_measures:
        fw = open(mean_imposter_path, "w")
        enroll_inner_means, _ = get_mean_std(
            enroll_embs, cohort_embs, top_n, cosine=False
        )
        test_inner_means, _ = get_mean_std(test_embs, cohort_embs, top_n, cosine=False)

    with open(score_file, "r") as fin, open(score_norm_file, "w") as fout:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            enroll_idx = enroll_utt2idx[line[0]]
            test_idx = test_utt2idx[line[1]]
            score = float(line[2])

            # AS-norm1 or S-norm
            normed_score = 0.5 * (
                (score - enroll_means[enroll_idx]) / enroll_stds[enroll_idx]
                + (score - test_means[test_idx]) / test_stds[test_idx]
            )
            if "imposter" in quality_measures:
                if enroll_inner_means[enroll_idx] > test_inner_means[test_idx]:
                    # enroll test low_quality high_quality
                    fw.write(
                        f"{line[0]} {line[1]} {enroll_inner_means[enroll_idx]} "
                        f"{test_inner_means[test_idx]}\n"
                    )
                else:
                    fw.write(
                        f"{line[0]} {line[1]} {test_inner_means[test_idx]} "
                        f"{enroll_inner_means[enroll_idx]}\n"
                    )

            # Z-norm or Adaptive Z-norm
            # normed_score = (score - enroll_means[enroll_idx]) / enroll_stds[enroll_idx]

            # T-norm or Adaptive T-norm
            # normed_score = (score - test_means[test_idx]) / test_stds[test_idx]
            fout.write(f"{line[0]} {line[1]} {normed_score:.5f}\n")
    if "imposter" in quality_measures:
        fw.close()

    logger.info("Score normalization is done.")

    return score_norm_file, mean_imposter_path
