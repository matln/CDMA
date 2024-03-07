#!/data/lijianchen/miniconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-
"""Define metric function (minDCF, EER)
Copyright 2021 Jianchen Li
"""

import logging
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def compute_eer_mindcf(scores, labels, Ptarget=0.01, c_miss=1, c_fa=1, eps=1e-6):
    """
    If score > threshold, prediction is positive, else negative

    scores : list
        similarity for target and non-target trials
    labels : list
        true labels for target and non-target trials

    return:
        eer: percent equal eeror rate (EER)
        dcf: minimum detection cost function (DCF) with voxceleb parameters

    Reference: Microsoft MSRTookit: compute_eer.m
               数据挖掘导论 p183
    """
    assert eps == 1e-6

    # Get the index list after sorting the scores list
    sorted_scores = [
        [index, value] for index, value in sorted(enumerate(scores), key=lambda x: x[1])
    ]
    # Sort the labels list
    sorted_labels = [labels[item[0]] for item in sorted_scores]
    sorted_labels = np.array(sorted_labels)

    FN = np.cumsum(sorted_labels == 1) / (sum(sorted_labels == 1) + eps)
    TN = np.cumsum(sorted_labels == 0) / (sum(sorted_labels == 0) + eps)
    FP = 1 - TN
    TP = 1 - FN

    FNR = FN / (TP + FN + eps)
    FPR = FP / (TN + FP + eps)
    difs = FNR - FPR
    idx1 = np.where(difs < 0, difs, float("-inf")).argmax(axis=0)
    idx2 = np.where(difs >= 0, difs, float("inf")).argmin(axis=0)
    # the x-axis of two points
    x = [FPR[idx1], FPR[idx2]]
    # the y-axis of two points
    y = [FNR[idx1], FNR[idx2]]
    z = [sorted_scores[idx1][1], sorted_scores[idx2][1]]
    # compute the intersection between the straight line connecting (x1, y1), (x2, y2)
    # and y = x.
    # Derivation: (x-x1) / (x2-x1) = (x-y1) / (y2-y1)                 ->
    #             (x-x1)(y2-y1) = (x-y1)(x2-x1)                       ->
    #             x(y2-y1-x2-x1) = x1(y2-y1) - y1(x2-x1)              ->
    #                            = x1(x2-x1) - y1(x2-x1)
    #                              + x1(y2-y1) - x1(x2-x1)            ->
    #                            = (x1-y1)(x2-x1) + x1(y2-y1-x2+x1)   ->
    #             x = x1 + (x1-y1)(x2-x1) / (y2-y1-x2-x1)
    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
    eer = 100 * (x[0] + a * (y[0] - x[0]))
    thresh1 = z[0] + (z[1] - z[0]) * (x[0] - y[0]) / (y[1] - x[1] - y[0] + x[0])

    # Compute dcf
    # VoxCeleb performance parameter
    # Cmiss = 1
    # Cfa = 1
    # avg = 0

    # # for Ptarget in [0.01, 0.001]:
    # for Ptarget in [0.01]:
    #     Cdet = Cmiss * FNR * Ptarget + Cfa * FPR * (1 - Ptarget)
    #     Cdef = min(Cmiss * Ptarget, Cfa * (1 - Ptarget))
    #     minDCF = min(Cdet) / Cdef
    #     avg += minDCF

    # # avg = avg / 2

    Cdet = c_miss * FNR * Ptarget + c_fa * FPR * (1 - Ptarget)
    Cdef = min(c_miss * Ptarget, c_fa * (1 - Ptarget))
    minDCF = min(Cdet) / Cdef
    idx = np.argmin(Cdet)
    fnr = FNR[idx]
    fpr = FPR[idx]
    thresh2 = sorted_scores[idx][1]

    return eer, thresh1, minDCF, thresh2, fnr, fpr


# --------------------------------------------------------------------------------------


def compute_eer_minDCF_sklearn(y_score, y, pos=1):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    # pos: 1 if higher is positive; 0 is lower is positive

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
    fnr = 1 - tpr
    print(len(fnr))
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    # Compute dcf
    # VoxCeleb performance parameter
    Cmiss = 1
    Cfa = 1
    avg = 0

    # for Ptarget in [0.01, 0.001]:
    for Ptarget in [0.01]:
        Cdet = Cmiss * fnr * Ptarget + Cfa * fpr * (1 - Ptarget)
        print(len(Cdet))
        Cdef = min(Cmiss * Ptarget, Cfa * (1 - Ptarget))
        minDCF = min(Cdet) / Cdef
        avg += minDCF

    return eer * 100, thresh


def compute_metric(
    score_file: str, trials: str, ptarget: float, return_thresh=False, output_to_file: bool = True
):
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

    eer, thresh1, minDCF, thresh2, fnr, fpr = compute_eer_mindcf(scores, labels, ptarget)
    # eer, threshold = compute_eer_minDCF_sklearn(scores, labels)

    if return_thresh:
        result = f"{eer:.3f} ({thresh1:.4f}) \u2502 {minDCF:.4f} ({thresh2:.4f})"
    else:
        result = f"{eer:.3f} \u2502 {minDCF:.4f}"
    if output_to_file:
        result_file = score_file.replace(".score", ".result")
        with open(result_file, "w") as fw:
            fw.write(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--score-file", type=str, default="", help="")
    parser.add_argument("--trials", type=str, default="", help="")
    parser.add_argument("--ptarget", type=float, default=0.01, help="")
    args = parser.parse_args()

    compute_metric(args.score_file, args.trials, args.ptarget, output_to_file=False)
