import os

cluster_path = "./gcn_gp/utt2cluster"

out_egs_dir = "exp/egs/voices_train_pseudo"
target_egs = "exp/egs/voices_train/train.egs.csv"

utt2clt = {}
with open(cluster_path, "r") as fr:
    lines = fr.readlines()
    for line in lines:
        utt, clt = line.strip().split()
        utt2clt[utt] = int(clt)
clts = sorted(list(set(utt2clt.values())))
clt2int = {}
for idx, clt in enumerate(clts):
    clt2int[clt] = idx
pseudo_num_spks = len(clts)

os.makedirs(f"{out_egs_dir}/info", exist_ok=True)
with open(f"{out_egs_dir}/train.egs.csv", "w") as fw:
    with open(target_egs, "r") as fr:
        lines = fr.readlines()
        fw.write(lines[0])
        for line in lines[1:]:
            line = line.strip().split(',')
            utt = line[0].rsplit("-", 1)[0]
            if utt in utt2clt:
                new_target = clt2int[utt2clt[utt]]
                line[-1] = str(new_target)
                fw.write(f"{','.join(line)}\n")

with open(f"{out_egs_dir}/info/num_targets", "w") as fr:
    fr.write(str(pseudo_num_spks))
