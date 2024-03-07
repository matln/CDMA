"""
Copyright 2019 alumae (https://github.com/alumae/sv_score_calibration)
          2022 Jianchen Li
"""
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .get_actdcf import compute_actDCF
from .get_eer_mindcf import compute_eer_mindcf

logger = logging.getLogger(__name__)


class LogisticRegression(nn.Module):
    def __init__(self, num_models, num_qualitys):
        super(LogisticRegression, self).__init__()
        self.fusion_layer = nn.Linear(num_models, 1)
        self.quality_layer = nn.Linear(num_qualitys, 1, bias=False)
        # TODO: 初始化
        nn.init.constant_(self.fusion_layer.weight, 1.0 / num_models)
        nn.init.constant_(self.fusion_layer.bias, 0)
        nn.init.constant_(self.quality_layer.weight, 0)

    def forward(self, scores, quals):
        out = self.fusion_layer(scores) + self.quality_layer(quals)
        return out


def cllr(tgt_llrs, ntgt_llrs):
    """
    Calculate the CLLR of the scores
    """

    def neg_log_sigmoid(lodds):
        """-log(sigmoid(log_odds))"""
        return torch.log1p(torch.exp(-lodds))

    return (
        0.5
        * (torch.mean(neg_log_sigmoid(tgt_llrs)) + torch.mean(neg_log_sigmoid(-ntgt_llrs)))
        / np.log(2)
    )


def load_score_files(score_files, trial2tgt=None):
    tgt_scores = None
    ntgt_scores = None
    scores = None
    for score_file in score_files:
        _tgt_scores = []
        _ntgt_scores = []
        _scores = []

        with open(score_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split()
                if trial2tgt is not None:
                    if trial2tgt[f"{line[0]}_{line[1]}"]:
                        _tgt_scores.append([float(line[2])])
                    else:
                        _ntgt_scores.append([float(line[2])])
                else:
                    _scores.append([float(line[2])])
        if trial2tgt is not None:
            _tgt_scores = torch.tensor(_tgt_scores, dtype=torch.float64)
            _ntgt_scores = torch.tensor(_ntgt_scores, dtype=torch.float64)

            if tgt_scores is None:
                tgt_scores = _tgt_scores
                ntgt_scores = _ntgt_scores
            else:
                tgt_scores = torch.cat((tgt_scores, _tgt_scores), dim=1)
                ntgt_scores = torch.cat((ntgt_scores, _ntgt_scores), dim=1)
        else:
            _scores = torch.tensor(_scores, dtype=torch.float64)

            if scores is None:
                scores = _scores
            else:
                scores = torch.cat((scores, _scores), dim=1)
    if trial2tgt is not None:
        return tgt_scores, ntgt_scores
    else:
        return scores


def load_quality_files(quality_files, trial2tgt=None):
    tgt_quals = None
    ntgt_quals = None
    quals = None

    for quality_file in quality_files:
        _tgt_quals = []
        _ntgt_quals = []
        _quals = []

        with open(quality_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split()
                trial = f"{line[0]}_{line[1]}"
                low_qual = float(line[2])
                high_qual = float(line[3])

                if trial2tgt is not None:
                    if trial2tgt[trial]:
                        _tgt_quals.append([low_qual, high_qual])
                    else:
                        _ntgt_quals.append([low_qual, high_qual])
                else:
                    _quals.append([low_qual, high_qual])

        if trial2tgt is not None:
            _tgt_quals = torch.tensor(_tgt_quals, dtype=torch.float64)
            _ntgt_quals = torch.tensor(_ntgt_quals, dtype=torch.float64)

            if tgt_quals is None:
                tgt_quals = _tgt_quals
                ntgt_quals = _ntgt_quals
            else:
                tgt_quals = torch.cat((tgt_quals, _tgt_quals), dim=1)
                ntgt_quals = torch.cat((ntgt_quals, _ntgt_quals), dim=1)
        else:
            _quals = torch.tensor(_quals, dtype=torch.float64)

            if quals is None:
                quals = _quals
            else:
                quals = torch.cat((quals, _quals), dim=1)
    if trial2tgt is not None:
        return tgt_quals, ntgt_quals
    else:
        return quals


def evalute_calibration(
    score_file: str, trials: str, ptarget: float, c_miss: int = 1, c_fa: int = 1
):
    scores = []
    labels = []
    trials2tgt = {}
    tgt_scores = []
    ntgt_scores = []

    with open(trials, "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            assert len(line) == 3
            if line[2] == "target":
                labels.append(1)
                trials2tgt[f"{line[0]}_{line[1]}"] = True
            elif line[2] == "nontarget":
                labels.append(0)
                trials2tgt[f"{line[0]}_{line[1]}"] = False

    with open(score_file, "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            assert len(line) == 3
            scores.append(float(line[2]))
            if trials2tgt[f"{line[0]}_{line[1]}"]:
                tgt_scores.append(float(line[2]))
            else:
                ntgt_scores.append(float(line[2]))

    actDCF, _, _ = compute_actDCF(np.array(scores), np.array(labels), ptarget, c_miss, c_fa)
    _, _, minDCF, _, _, _ = compute_eer_mindcf(scores, labels, ptarget, c_miss, c_fa)
    cllr_value = cllr(
        torch.tensor(tgt_scores, dtype=torch.float64),
        torch.tensor(ntgt_scores, dtype=torch.float64),
    )
    return minDCF, actDCF, cllr_value


def calibrate(
    trials_file: str,
    dev_score_files: list,
    eval_score_files: list,
    dev_quality_files: list = None,
    eval_quality_files: list = None,
    out_score_file: str = None,
    max_epochs: int = 100,
):
    trial2tgt = {}
    with open(trials_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line[2] == "target":
                is_target = True
            elif line[2] == "nontarget":
                is_target = False
            else:
                raise ValueError
            trial2tgt[f"{line[0]}_{line[1]}"] = is_target

    tgt_scores, ntgt_scores = load_score_files(dev_score_files, trial2tgt)

    if dev_quality_files is not None:
        assert eval_quality_files is not None
        tgt_quals, ntgt_quals = load_quality_files(dev_quality_files, trial2tgt)

    model = LogisticRegression(tgt_scores.size(1), tgt_quals.size(1))
    model.double()

    optimizer = optim.LBFGS(model.parameters(), lr=0.01)

    best_loss = 1000000.0

    for i in range(max_epochs):

        def closure():
            optimizer.zero_grad()
            ntgt_llrs = model(ntgt_scores, ntgt_quals)
            tgt_llrs = model(tgt_scores, tgt_quals)
            loss = cllr(tgt_llrs, ntgt_llrs)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if best_loss - loss < 1e-6:
            break
        else:
            if loss < best_loss:
                best_loss = loss

    logger.info(f"Quality weights: {model.quality_layer.weight.detach().numpy()[0]}")

    # Apply the calibration model to eval scores
    model.eval()

    scores = load_score_files(eval_score_files)

    if eval_quality_files is not None:
        assert dev_quality_files is not None
        quals = load_quality_files(eval_quality_files)

    llrs = model(scores, quals)
    if out_score_file is None:
        out_score_file = eval_score_files[0].replace(".score", "_qmf.score")
    with open(out_score_file, "w") as f_out, open(eval_score_files[0], "r") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split()
            f_out.write(f"{line[0]} {line[1]} {llrs[i].cpu().item()}\n")
    return out_score_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calibrates speaker verification LLR scores")
    parser.add_argument("--max-epochs", default=100)
    parser.add_argument(
        "--trials-file",
        type=str,
        default="",
        help="Speaker recognition trials file. Each line is a \n"
        "triple <enroll_utt> <test_utt> target|nontarget",
    )
    parser.add_argument(
        "--dev-score-files",
        type=str,
        default="",
        help="One or more score files. Each line is a \n" "triple <enroll_utt> <test_utt> <score>",
    )
    parser.add_argument(
        "--eval-score-files",
        type=str,
        default="",
        help="One or more score files. Each line is a \n" "triple <enroll_utt> <test_utt> <score>",
    )
    parser.add_argument(
        "--dev-quality-files",
        type=str,
        default="",
        help="None or multiple quality files. Each line is \n"
        "<enroll_utt> <test_utt> <low_quality> <high_quality>",
    )
    parser.add_argument(
        "--eval-quality-files",
        type=str,
        default="",
        help="None or multiple quality files. Each line is \n"
        "<enroll_utt> <test_utt> <low_quality> <high_quality>",
    )
    args = parser.parse_args()

    calibrate(
        args.trials_file,
        args.dev_score_files.split(","),
        args.eval_score_files.split(","),
        args.dev_quality_files.split(","),
        args.eval_quality_files.split(","),
        args.out_score_file,
    )
