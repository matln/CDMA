import os
import torch
import logging
import numpy as np
from build_knn import GCNDataset
from scipy.sparse import csr_matrix
from models import HEAD, HEAD_test, GCN, FC
from sklearn.metrics import normalized_mutual_info_score
from rich import print as pprint
from collections import Counter

import sys

sys.path.insert(0, "./")

from metrics import bcubed
from speakernet.utils.rich_utils import track
from speakernet.utils.utils import Timer
from speakernet.utils.logging_utils import init_logger

logger = init_logger()
logger.setLevel(logging.ERROR)

# logger.setLevel(logging.ERROR)
# xvector_path = "exp/pretrain/2022-11-19_17:34:35/embeddings/near_epoch_19/voices_val/xvector.scp"
xvector_path = "exp/pretrain/2022-11-19_17:34:35/embeddings/near_epoch_19/voices_train/xvector.scp"

k1 = 40
k2 = 80
MODEL_ROOT = "gcn_gp/models/gcn_2layers"
knn_path1 = f"gcn_gp/knns/voices_k{k1}.npz"
knn_path2 = f"gcn_gp/knns/voices_k{k2}.npz"

torch.cuda.set_device(0)


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while u != parent[u]:
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


def main(epoch, lambda1, lambda2):
    model_path = f"{MODEL_ROOT}/Backbone_Epoch_{epoch}.pth"
    head_path = f"{MODEL_ROOT}/Head_Epoch_{epoch}.pth"
    logger.info(f"knn_path1: {knn_path1}, model_path: {model_path}, head_path: {head_path}")
    pprint(f"knn_path1: {knn_path1}, model_path: {model_path}, head_path: {head_path}")
    pprint(k1, k2)

    dataset1 = GCNDataset(
        xvector_path=xvector_path,
        k=k1,
        knn_path=knn_path1,
        force=True,
    )
    knns1 = dataset1.knns
    utts = dataset1.utts
    features1 = torch.FloatTensor(dataset1.features)
    adj1 = dataset1.adj
    gt_labels1 = dataset1.gt_labels

    dataset2 = GCNDataset(
        xvector_path=xvector_path,
        k=k2,
        knn_path=knn_path2,
        force=True,
    )
    knns2 = dataset2.knns

    n = len(knns2)
    nbrs = knns2[:, 0, :]
    edges = []
    score = []
    inst_num = knns2.shape[0]
    k_num = knns2.shape[2]
    logger.info(f"inst_num: {inst_num}")

    # print(**cfg.model['kwargs'])
    model = GCN(feature_dim=256, nhid=512, dropout=0)
    # model = FC(feature_dim=256, nhid=512, dropout=0)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    HEAD_test1 = HEAD_test(nhid=512)
    HEAD_test1.load_state_dict(torch.load(head_path, map_location="cpu"))

    pair_a = []
    pair_b = []
    pair_a_new = []
    pair_b_new = []
    for i in range(inst_num):
        pair_a.extend([int(i)] * k_num)
        pair_b.extend([int(j) for j in nbrs[i]])
    for i in range(len(pair_a)):
        if pair_a[i] != pair_b[i]:
            pair_a_new.extend([pair_a[i]])
            pair_b_new.extend([pair_b[i]])
    pair_a = pair_a_new
    pair_b = pair_b_new
    # pprint(len(pair_a))
    inst_num = len(pair_a)

    model.cuda()
    HEAD_test1.cuda()
    features = features1.cuda()
    adj = adj1.cuda()
    labels = torch.from_numpy(gt_labels1).cuda()

    model.eval()
    HEAD_test1.eval()
    test_data = [features, adj, labels]

    disconnect_num = 0
    positive_num = 0
    negative_num = 0
    TP_num = 0
    TN_num = 0
    edges = None
    for threshold1 in [lambda1]:
        with torch.no_grad():
            output_feature = model(test_data)

            patch_num = 11
            patch_size = int(inst_num / patch_num)
            for i in track(range(patch_num)):
            # for i in range(patch_num):
                id1 = pair_a[i * patch_size : (i + 1) * patch_size]
                id2 = pair_b[i * patch_size : (i + 1) * patch_size]
                score_ = HEAD_test1(output_feature[id1], output_feature[id2])
                score_ = np.array(score_)
                idx = np.where(score_ > threshold1)[0].tolist()

                patch_label = (labels[id1] == labels[id2]).long()
                positive_idx = torch.nonzero(patch_label).squeeze(1).tolist()
                negative_idx = torch.nonzero(patch_label - 1).squeeze(1).tolist()
                TP_num += len(np.where(score_[positive_idx] > threshold1)[0].tolist())
                TN_num += len(np.where(score_[negative_idx] < threshold1)[0].tolist())

                # idx = positive_idx

                negative_num += len(negative_idx)
                positive_num += len(positive_idx)
                disconnect_num += len(score_) - len(idx)

                # score.extend(score_[idx].tolist())
                id1 = np.array(id1)
                id2 = np.array(id2)
                id1 = np.array([id1[idx].tolist()])
                id2 = np.array([id2[idx].tolist()])
                if edges is None:
                    edges = np.concatenate([id1, id2], 0).transpose()
                else:
                    if len(idx) != 0:
                        edges = np.concatenate([edges, np.concatenate([id1, id2], 0).transpose()], 0)

            id1 = pair_a[(i + 1) * patch_size :]
            if len(id1) != 0:
                id2 = pair_b[(i + 1) * patch_size :]
                score_ = HEAD_test1(output_feature[id1], output_feature[id2])
                score_ = np.array(score_)
                idx = np.where(score_ > threshold1)[0].tolist()
                # score.extend(score_[idx].tolist())
                patch_label = (labels[id1] == labels[id2]).long()
                positive_idx = torch.nonzero(patch_label).squeeze(1).tolist()
                negative_idx = torch.nonzero(patch_label - 1).squeeze(1).tolist()
                TP_num += len(np.where(score_[positive_idx] > threshold1)[0].tolist())
                TN_num += len(np.where(score_[negative_idx] < threshold1)[0].tolist())

                negative_num += len(negative_idx)
                positive_num += len(positive_idx)
                disconnect_num += len(score_) - len(idx)
                id1 = np.array(id1)
                id2 = np.array(id2)
                id1 = np.array([id1[idx].tolist()])
                id2 = np.array([id2[idx].tolist()])
                if len(idx) != 0:
                    edges = np.concatenate([edges, np.concatenate([id1, id2], 0).transpose()], 0)

        value = [1] * len(edges)

        # with Timer("csr_matrix"):
        if len(edges) == 0:
            edges = np.array([[0, 0]])
            value = [1] * len(edges)
        adj2 = csr_matrix((value, (edges[:, 0].tolist(), edges[:, 1].tolist())), shape=(n, n),)
        link_num = np.array(adj2.sum(axis=1))
        common_link = adj2.dot(adj2)

        # with Timer("edge_to_connected_graph"):
        labels1 = edge_to_connected_graph(edges.tolist(), n)

        for threshold2 in [lambda2]:
            edges_new = []
            edges = np.array(edges)
            share_num = common_link[edges[:, 0].tolist(), edges[:, 1].tolist()].reshape(-1, 1)
            jaccard = share_num / (link_num[edges[:, 0]] + link_num[edges[:, 1]] - share_num + 1e-8)
            jaccard = np.squeeze(np.array(jaccard))
            edges_new = edges[jaccard >= threshold2].tolist()

            # with Timer("edge_to_connected_graph"):
            labels2 = edge_to_connected_graph(edges_new, n)
            pprint(f"# the threshold1 is: {threshold1}, the threshold2 is: {threshold2}")
            nmi_2 = normalized_mutual_info_score(gt_labels1, labels2)
            b_pre, b_rec, b_fscore_2 = bcubed(gt_labels1, labels2)
            pprint("Before filtering:")
            pprint(f"total clusters: {max(labels2)}, nmi_2: {nmi_2:.3f}, "
                  f"b_fscore: {b_fscore_2:.3f}, b_pre: {b_pre:.3f}, b_rec: {b_rec:.3f}")
            counter_lbl = Counter(labels2)
            sorted_counter = sorted(counter_lbl.items(), key=lambda x: x[1], reverse=True)
            # print(dict(sorted_counter))


            # Filter
            clt2idx = {}
            for i, clt in enumerate(labels2):
                clt2idx.setdefault(clt, []).append(i)
            remain_idx = [v for k, v in clt2idx.items() if len(v) >= 10]
            remain_idx = [x for sub_lst in remain_idx for x in sub_lst]
            remain_labels2 = labels2[remain_idx]
            remain_gt_labels1 = gt_labels1[remain_idx]
            nmi_3 = normalized_mutual_info_score(remain_gt_labels1, remain_labels2)
            b_pre, b_rec, b_fscore_3 = bcubed(remain_gt_labels1, remain_labels2)
            pprint("After filtering:")
            pprint(f"total clusters: {len(list(set(remain_labels2)))}, nmi_3: {nmi_3:.3f}, "
                  f"b_fscore: {b_fscore_3:.3f}, b_pre: {b_pre:.3f}, b_rec: {b_rec:.3f}")
            counter_lbl = Counter(remain_labels2)
            sorted_counter = sorted(counter_lbl.items(), key=lambda x: x[1], reverse=True)
            print(f"remain: {len(remain_labels2)}, total: {len(labels2)}")
            print(dict(sorted_counter[:10]))

            cluster2lbl = {}
            lbl = 0
            with open("gcn_gp/utt2cluster", "w") as fw:
                for idx in remain_idx:
                    utt = utts[idx]
                    _cluster = labels2[idx]
                    if _cluster not in cluster2lbl:
                        cluster2lbl[_cluster] = lbl
                        lbl += 1
                    fw.write(f"{utt} {cluster2lbl[_cluster]}\n")



if __name__ == "__main__":
    main("5", 0.75, 0.25)
