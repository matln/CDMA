"""
Copyright 2022 Jianchen Li
"""

import argparse
import numpy as np


def compute_actDCF(
    scores: np.ndarray, labels: np.ndarray, p_target: float, c_miss: int = 1, c_fa: int = 1
):
    # Bayesian decision threshold
    beta = c_fa * (1 - p_target) / (c_miss * p_target)
    decisions = (scores >= np.log(beta)).astype("i")
    num_targets = np.sum(labels)
    fp = np.sum(decisions * (1 - labels))
    num_nontargets = np.sum(1 - labels)
    fn = np.sum((1 - decisions) * labels)
    fpr = fp / num_nontargets if num_nontargets > 0 else np.nan
    fnr = fn / num_targets if num_targets > 0 else np.nan
    actDCF = fnr + beta * fpr
    return actDCF, fpr, fnr


def compute_metric(score_file: str, trials: str, ptarget: float, c_miss: int = 1, c_fa: int = 1):
    scores = []
    labels = []

    with open(score_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(" ")
            assert len(line) == 3
            scores.append(float(line[2]))

    with open(trials, "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(" ")
            assert len(line) == 3
            if line[2] == "target":
                labels.append(1)
            elif line[2] == "nontarget":
                labels.append(0)

    actDCF, fnr, fpr = compute_actDCF(scores, labels, ptarget, c_miss, c_fa)
    # eer, threshold = compute_eer_minDCF_sklearn(scores, labels)

    result = f"{actDCF:.3f} \u2502 {actDCF:.4f}"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--score-file", type=str, default="", help="")
    parser.add_argument("--trials", type=str, default="", help="")
    parser.add_argument("--ptarget", type=float, default=0.01, help="")
    args = parser.parse_args()

    compute_metric(args.score_file, args.trials, args.ptarget, c_miss=1, c_fa=1)
