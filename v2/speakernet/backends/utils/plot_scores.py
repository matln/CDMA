"""
Copyright 2022 Jianchen Li
"""

import sys
import warnings
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

sys.path.insert(0, "./")
from speakernet.backends.get_eer_mindcf import compute_eer_mindcf

warnings.filterwarnings("ignore")


def plot_score(scores_file, trials):
    keys2score = {}
    with open(scores_file, "r") as r:
        lines = r.readlines()
        for line in lines:
            enroll, test, score = line.strip().split()
            keys2score[enroll + " " + test] = score

    target_scores = []
    nontarget_scores = []
    scores = []
    labels = []
    with open(trials, "r") as r:
        lines = r.readlines()
        for line in lines:
            enroll, test, target = line.strip().split()
            scores.append(float(keys2score[enroll + " " + test]))
            if target == "target":
                target_scores.append(float(keys2score[enroll + " " + test]))
                labels.append(1)
            elif target == "nontarget":
                nontarget_scores.append(float(keys2score[enroll + " " + test]))
                labels.append(0)

    eer, thresh1, minDCF, thresh2, FNR, FPR = compute_eer_mindcf(scores, labels)
    print(eer, thresh1, minDCF, thresh2)

    sns.distplot(target_scores, bins="auto", fit=norm, kde=False, label="target")
    sns.distplot(nontarget_scores, bins="auto", fit=norm, kde=False, label="nontarget")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.title(
        "eer: {:.3f}, minDCF: {:.4f}, thr_eer: {:.3f}, thr_minDCF: {:.3f}".format(
            eer, minDCF, thresh1, thresh2
        )
    )
    plt.legend()
    plt.scatter(thresh1, 0, marker="o", s=100)
    plt.scatter(thresh2, 0, marker="o", s=100)

    plt.savefig(
        "{}/epoch{}.png".format("/data/lijianchen", 3.1),
        bbox_inches="tight",
        dpi=800,
        pad_inches=0.0,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.RawTextHelpFormatter, conflict_handler="resolve"
    )
    parser.add_argument("--scores", type=str, default="", help="")
    parser.add_argument("--trials", type=str, default="", help="")
    args = parser.parse_args()

    plot_score(args.scores, args.trials)
