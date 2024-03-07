import os
import sys
import torch
import random
import logging
import numpy as np
import torch.optim as optim
from rich import print
from torch.utils.data import Dataset, DataLoader, Sampler

sys.path.insert(0, "./")
from models import HEAD, GCN
from build_knn import GCNDataset
import speakernet.utils.utils as utils
from speakernet.utils.rich_utils import track
from speakernet.utils.logging_utils import init_logger


logger = init_logger()

total_epochs = 6
seed = 1024
k1 = 80
k2 = 80
MODEL_ROOT = f"gcn_gp/models/gcn_2layers"
xvector_path = "exp/pretrain/2022-11-19_17:34:35/embeddings/near_epoch_19/voxceleb2_dev_subset/xvector.scp"
knn_path1 = f"gcn_gp/knns/vox_k{k1}.npz"
knn_path2 = f"gcn_gp/knns/vox_k{k2}.npz"

torch.cuda.set_device(0)
utils.set_seed(1024, deterministic=True)


def main():
    # prepare dataset
    logger.info("Loading the training data...")
    dataset = GCNDataset(
        xvector_path=xvector_path,
        k=k1,
        knn_path=knn_path1,
        force=False,
    )
    features = torch.FloatTensor(dataset.features)
    labels = torch.LongTensor(dataset.gt_labels)
    adj = dataset.adj
    logger.info("Have loaded the training data.")

    logger.info("Loading the larger knns ...")
    dataset2 = GCNDataset(
        xvector_path=xvector_path,
        k=k2,
        knn_path=knn_path2,
        force=False,
    )
    adj2 = dataset2.adj

    feature_dim = dataset.feature_dim
    assert feature_dim == 256

    model = GCN(feature_dim=feature_dim, nhid=512, dropout=0)
    HEAD1 = HEAD(nhid=512)

    optimizer = optim.SGD(
        [
            {"params": model.parameters(), "weight_decay": 1e-5},
            {"params": HEAD1.parameters(), "weight_decay": 1e-5},
        ],
        lr=0.01,
        momentum=0.9,
    )
    logger.info("the learning rate is 0.01")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(total_epochs * 0.5), int(total_epochs * 0.8), int(total_epochs * 0.9)],
        gamma=0.1,
    )

    model = model.cuda()
    HEAD1 = HEAD1.cuda()
    # 1. put selected feature and labels to cuda
    labels = labels.cuda()
    features = features.cuda()
    adj = adj.cuda()
    train_data = [features, adj, labels]

    logger.info(f"the model save path is {MODEL_ROOT}")

    for epoch in range(total_epochs):
        model.train()
        HEAD1.train()

        # 2. train the model
        train_id_inst = adj2._indices().size()[1]
        logger.info(f"train_id_inst: {train_id_inst}")

        rad_id = torch.randperm(train_id_inst).tolist()
        patch_num = 640
        for i in track(range(patch_num)):
            optimizer.zero_grad()

            id = rad_id[
                i * int(train_id_inst / patch_num) : (i + 1) * int(train_id_inst / patch_num)
            ]
            x = model(train_data)
            loss, pos_acc, neg_acc = HEAD1(x, adj2, labels, id)

            loss.backward()
            optimizer.step()

            print(
                "epoch: {}/{}, patch: {}/{}, loss: {:.4f}, pos_acc: {:.4f}, neg_acc: {:.4f}".format(
                    epoch + 1, total_epochs, i, patch_num, loss, pos_acc, neg_acc
                )
            )

        # lr_scheduler.step(epoch)

        # 3. save model
        if not os.path.exists(MODEL_ROOT):
            os.makedirs(MODEL_ROOT)
        logger.info("save model in epoch:{} to {}".format(epoch, MODEL_ROOT))
        torch.save(
            model.state_dict(), os.path.join(MODEL_ROOT, "Backbone_Epoch_{}.pth".format(epoch + 1)),
        )
        torch.save(
            HEAD1.state_dict(), os.path.join(MODEL_ROOT, "Head_Epoch_{}.pth".format(epoch + 1)),
        )


if __name__ == "__main__":
    main()
