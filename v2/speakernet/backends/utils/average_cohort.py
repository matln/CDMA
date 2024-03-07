"""
Copyright 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
          2022 Jianchen Li
"""

import os
import kaldiio
import numpy as np

from speakernet.utils.utils import read_scp


def average_spk_emb(spk2utt_file, xvector_scp, spk_xvector_scp):
    # read spk2utt
    spk2utt = read_scp(spk2utt_file, multi_value=True)

    utt2emb = {}
    for utt, emb in kaldiio.load_scp_sequential(xvector_scp):
        # emb = emb / np.sqrt(np.sum(emb ** 2))
        utt2emb[utt] = emb

    dir_name = os.path.dirname(spk_xvector_scp)
    os.makedirs(dir_name, exist_ok=True)

    spk_xvector_scp = os.path.abspath(spk_xvector_scp)
    spk_xvector_ark = spk_xvector_scp.replace("scp", "ark")
    with kaldiio.WriteHelper(f"ark,scp:{spk_xvector_ark},{spk_xvector_scp}") as writer:
        for spk in spk2utt.keys():
            utts = spk2utt[spk]
            mean_emb = None
            utt_num = 0
            for utt in utts:
                emb = utt2emb[utt]
                if mean_emb is None:
                    mean_emb = np.zeros_like(emb)
                mean_emb += emb
                utt_num += 1
            mean_emb = mean_emb / utt_num
            writer(spk, mean_emb)
