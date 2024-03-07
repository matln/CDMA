import os
import kaldiio
import faiss
import time
import torch
import logging
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, "./")

from speakernet.utils.rich_utils import track
from speakernet.utils.utils import Timer
from rich import print

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)

eps = 1e-2
th_sim = 0.


class GCNDataset(object):
    def __init__(self, xvector_path, k, knn_path, force=False, use_sim = True, weighted=True):
        # 1. Load features and labels
        root_logger.info("Load features and labels...")
        features = []
        spk2label = {}
        gt_labels = []
        utts = []
        num = 0
        # for idx, (utt, feat) in enumerate(track(kaldiio.load_scp_sequential(xvector_path))):
        for idx, (utt, feat) in enumerate(kaldiio.load_scp_sequential(xvector_path)):
            spk = utt.split("-")[0]
            if spk not in spk2label:
                spk2label[spk] = num
                num += 1
            gt_labels.append(spk2label[spk])
            utts.append(utt)
            features.append(feat)
        _features = np.array(features)
        self.features = _features
        features = _features / np.linalg.norm(_features, axis=1).reshape(-1, 1)
        self.feature_dim = self.features.shape[1]
        self.gt_labels = np.array(gt_labels)
        self.inst_num = len(features)
        self.utts = utts

        # 2. Build knns
        if not os.path.exists(knn_path) or force:
            root_logger.info("Build knns...")
            features = features.astype("float32")
            size, dim = features.shape
            index = faiss.IndexFlatIP(dim)
            index.add(features)
            with Timer():
                sims, nbrs = index.search(features, k=k)
            if weighted:
                knns = [
                    (
                        np.array(nbr, dtype=np.int32),
                        1 - np.minimum(np.maximum(np.array(sim, dtype=np.float32), 0), 1),
                    )
                    for nbr, sim in zip(nbrs, sims)
                ]
            else:
                knns = [
                    (
                        np.array(nbr, dtype=np.int32),
                        np.ones_like(np.array(sim, dtype=np.float32)),
                    )
                    for nbr, sim in zip(nbrs, sims)
                ]
                use_sim = False
            # Save knns
            root_logger.info("Save knns...")
            np.savez_compressed(knn_path, data=knns)
        else:
            # Read knns
            root_logger.info("Read knns...")
            knns = np.load(knn_path, allow_pickle=True)['data']
        self.knns = np.array(knns)

        # 3. Convert knns to sparse matrix
        root_logger.info("Convert knns to sparse matrix...")
        n = len(knns)
        if isinstance(knns, list):
            knns = np.array(knns)
        nbrs = knns[:, 0, :]
        self.nbrs = nbrs
        dists = knns[:, 1, :]
        assert -eps <= dists.min() <= dists.max() <= 1 + eps, "min: {}, max: {}".format(dists.min(), dists.max())
        if use_sim:
            sims = 1. - dists
        else:
            sims = dists
        row, col = np.where(sims >= th_sim)
        # remove the self-loop
        idxs = np.where(row != nbrs[row, col])
        row = row[idxs]
        col = col[idxs]
        data = sims[row, col]
        col = nbrs[row, col]  # convert to absolute column
        self.row = row
        self.col = col
        assert len(row) == len(col) == len(data)
        spmat = csr_matrix((data, (row, col)), shape=(n, n))
        # mat = spmat.toarray()
        # print(spmat.shape)
        # print(spmat.getnnz())

        # 4. Build symmetric adjacency matrix
        root_logger.info("Build symmetric adjacency matrix...")
        adj = spmat
        ## adj.multiply(adj.T > adj) is none
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # print(adj.getnnz())
        ## self loop:
        self._adj = adj
        adj = adj + sp.eye(adj.shape[0])
        # print(adj.getnnz())

        # 5. Row-normalize sparse matrix
        root_logger.info("Row-normalize sparse matrix...")
        rowsum = np.array(adj.sum(1))
        # if rowsum <= 0, keep its previous value
        rowsum[rowsum <= 0] = 1
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj = r_mat_inv.dot(adj)

        # 6. Convert sparse matrix to indices values
        root_logger.info("Convert sparse matrix to indices values...")
        sparse_mx = adj.tocoo().astype(np.float32)
        # print(sparse_mx)
        self.adj_indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        self.adj_values = sparse_mx.data
        self.adj_shape = np.array(sparse_mx.shape)

        indices = torch.from_numpy(self.adj_indices)
        values = torch.from_numpy(self.adj_values)
        shape = torch.Size(self.adj_shape)
        self.adj = torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    dataset = GCNDataset()
