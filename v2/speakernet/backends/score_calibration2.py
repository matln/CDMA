"""
Copyright 2019 alumae (https://github.com/alumae/sv_score_calibration)
          2022 Jianchen Li
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim


# https://pytorch.org/tutorials/intermediate/parametrizations.html
def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)


class LinearSymmetric(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(input_dim, input_dim))

    def forward(self, enroll_quals, test_quals):
        W = symmetric(self.weight)
        return torch.diag(enroll_quals @ W @ test_quals.T).unsqueeze(1)


class LogisticRegression(nn.Module):
    def __init__(self, num_models, num_qualitys):
        super(LogisticRegression, self).__init__()
        self.fusion_layer = nn.Linear(num_models, 1)
        self.quality_layer = LinearSymmetric(num_qualitys)
        # self.quality_layer = nn.Linear(num_qualitys, 1, bias=False)
        # TODO: 初始化
        nn.init.constant_(self.fusion_layer.weight, 1.0 / (num_models))
        nn.init.constant_(self.fusion_layer.bias, 0)
        nn.init.constant_(self.quality_layer.weight, 0)

    def forward(self, scores, enroll_quals, test_quals):
        out = self.fusion_layer(scores) + self.quality_layer(enroll_quals, test_quals)
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
                        _tgt_scores.append(float(line[2]))
                    else:
                        _ntgt_scores.append(float(line[2]))
                else:
                    _scores.append(float(line[2]))
        if trial2tgt is not None:
            _tgt_scores = torch.tensor(_tgt_scores, dtype=torch.float64).unsqueeze(1)
            _ntgt_scores = torch.tensor(_ntgt_scores, dtype=torch.float64).unsqueeze(1)

            if tgt_scores is None:
                tgt_scores = _tgt_scores
                ntgt_scores = _ntgt_scores
            else:
                tgt_scores = torch.cat((tgt_scores, _tgt_scores), dim=1)
                ntgt_scores = torch.cat((ntgt_scores, _ntgt_scores), dim=1)
            return tgt_scores, ntgt_scores
        else:
            _scores = torch.tensor(_scores, dtype=torch.float64).unsqueeze(1)

            if scores is None:
                scores = _scores
            else:
                scores = torch.cat((scores, _scores), dim=1)
            return scores


def load_quality_files(quality_files, trial2tgt=None):
    tgt_enroll_quals = None
    tgt_test_quals = None
    ntgt_enroll_quals = None
    ntgt_test_quals = None
    enroll_quals = None
    test_quals = None

    for quality_file in quality_files:
        _tgt_enroll_quals = []
        _tgt_test_quals = []
        _ntgt_enroll_quals = []
        _ntgt_test_quals = []
        _enroll_quals = []
        _test_quals = []

        with open(quality_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split()
                trial = f"{line[0]}_{line[1]}"
                enroll_qual = float(line[2])
                test_qual = float(line[3])

                if trial2tgt is not None:
                    if trial2tgt[trial]:
                        _tgt_enroll_quals.append(enroll_qual)
                        _tgt_test_quals.append(test_qual)
                    else:
                        _ntgt_enroll_quals.append(enroll_qual)
                        _ntgt_test_quals.append(test_qual)
                else:
                    _enroll_quals.append(enroll_qual)
                    _test_quals.append(test_qual)

        if trial2tgt is not None:
            _tgt_enroll_quals = torch.tensor(_tgt_enroll_quals, dtype=torch.float64).unsqueeze(1)
            _tgt_test_quals = torch.tensor(_tgt_test_quals, dtype=torch.float64).unsqueeze(1)
            _ntgt_enroll_quals = torch.tensor(_ntgt_enroll_quals, dtype=torch.float64).unsqueeze(1)
            _ntgt_test_quals = torch.tensor(_ntgt_test_quals, dtype=torch.float64).unsqueeze(1)

            if tgt_enroll_quals is None:
                tgt_enroll_quals = _tgt_enroll_quals
                tgt_test_quals = _tgt_test_quals
                ntgt_enroll_quals = _ntgt_enroll_quals
                ntgt_test_quals = _ntgt_test_quals
            else:
                tgt_enroll_quals = torch.cat((tgt_enroll_quals, _tgt_enroll_quals), dim=1)
                tgt_test_quals = torch.cat((tgt_test_quals, _tgt_test_quals), dim=1)
                ntgt_enroll_quals = torch.cat((ntgt_enroll_quals, _ntgt_enroll_quals), dim=1)
                ntgt_test_quals = torch.cat((ntgt_test_quals, _ntgt_test_quals), dim=1)
            return tgt_enroll_quals, tgt_test_quals, ntgt_enroll_quals, ntgt_test_quals
        else:
            _enroll_quals = torch.tensor(_enroll_quals, dtype=torch.float64).unsqueeze(1)
            _test_quals = torch.tensor(_test_quals, dtype=torch.float64).unsqueeze(1)

            if enroll_quals is None:
                enroll_quals = _enroll_quals
                test_quals = _test_quals
            else:
                enroll_quals = torch.cat((enroll_quals, _enroll_quals), dim=1)
                test_quals = torch.cat((test_quals, _test_quals), dim=1)
            return enroll_quals, test_quals


def calibrate(
    trials_file: str,
    dev_score_files: list,
    eval_score_files: list,
    out_score_file: str,
    dev_quality_files: list = None,
    eval_quality_files: list = None,
    max_epochs: int = 50,
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
        tgt_enroll_quals, tgt_test_quals, ntgt_enroll_quals, ntgt_test_quals = load_quality_files(
            dev_quality_files, trial2tgt
        )

    model = LogisticRegression(tgt_scores.size(1), tgt_enroll_quals.size(1))
    model.double()
    model.to(torch.device("cuda:1"))

    optimizer = optim.LBFGS(model.parameters(), lr=0.005)

    best_loss = 1000000.0

    for i in range(max_epochs):
        print("STEP: ", i)

        def closure():
            optimizer.zero_grad()
            ntgt_llrs = model(ntgt_scores.to(torch.device("cuda:1")), ntgt_enroll_quals.to(torch.device("cuda:1")), ntgt_test_quals.to(torch.device("cuda:1")))
            tgt_llrs = model(tgt_scores.to(torch.device("cuda:1")), tgt_enroll_quals.to(torch.device("cuda:1")), tgt_test_quals.to(torch.device("cuda:1")))
            loss = cllr(tgt_llrs, ntgt_llrs)
            print("  loss:", loss.item())
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        print("  loss:", loss.item())
        if best_loss - loss < 1e-6:
            print("Converged!")
            break
        else:
            if loss < best_loss:
                best_loss = loss

    # Apply the calibration model to eval scores
    model.eval()

    scores = load_score_files(eval_score_files)

    if eval_quality_files is not None:
        assert dev_quality_files is not None
        enroll_quals, test_quals = load_quality_files(
            eval_quality_files
        )

    llrs = model(scores.to(torch.device("cuda:1")), enroll_quals.to(torch.device("cuda:1")), test_quals.to(torch.device("cuda:1")))
    with open(out_score_file, "w") as f_out, open(eval_score_files[0], "r") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split()
            f_out.write(f"{line[0]} {line[1]} {llrs[i].cpu().item()}\n")


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
        "<enroll_utt> <test_utt> <enroll_quality> <test_quality>",
    )
    parser.add_argument(
        "--eval-quality-files",
        type=str,
        default="",
        help="None or multiple quality files. Each line is \n"
        "<enroll_utt> <test_utt> <enroll_quality> <test_quality>",
    )
    parser.add_argument("--out-score-file", type=str, default="", help="")
    args = parser.parse_args()
    
    calibrate(args.trials_file, args.dev_score_files.split(","), args.eval_score_files.split(","), args.out_score_file, args.dev_quality_files.split(","), args.eval_quality_files.split(","))
