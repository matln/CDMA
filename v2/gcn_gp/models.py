import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_acc(outs, labels):
    preds = torch.squeeze(torch.argmax(outs.detach(), dim=1))
    num_correct = (labels == preds).sum()
    acc = num_correct.item() / len(labels)
    return acc
    # return num_correct.item()


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        if features.dim() == 2:
            x = torch.spmm(A, features)
        elif features.dim() == 3:
            x = torch.bmm(A, features)
        else:
            raise RuntimeError("the dimension of features should be 2 or 3")
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg, dropout=0):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)
        self.agg = agg()
        self.dropout = dropout

    def forward(self, features, A):
        feat_dim = features.shape[-1]
        assert feat_dim == self.in_dim
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=-1)
        if features.dim() == 2:
            op = "nd,df->nf"
        elif features.dim() == 3:
            op = "bnd,df->bnf"
        else:
            raise RuntimeError("the dimension of features should be 2 or 3")
        out = torch.einsum(op, (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training=self.training)
        return out


class GCN(nn.Module):
    def __init__(self, feature_dim, nhid, dropout=0):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, nhid, MeanAggregator, dropout)
        # self.conv3 = GraphConv(nhid, nhid, MeanAggregator, dropout)

    def forward(self, data):
        x, adj = data[0], data[1]

        x = self.conv1(x, adj)
        x = F.relu(self.conv2(x, adj) + x)
        # x = F.relu(self.conv3(x, adj) + x)

        return x


class FC(nn.Module):
    def __init__(self, feature_dim, nhid, dropout=0):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(feature_dim, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, adj = data[0], data[1]

        x = self.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x)
        # x = F.relu(self.conv3(x, adj) + x)

        return x


class HEAD(nn.Module):
    def __init__(self, nhid, dropout=0):
        super(HEAD, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(
            nn.Linear(nhid * 2, nhid), nn.PReLU(nhid), nn.Linear(nhid, 2)
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, feature, adj, label, select_id=None):
        feature = feature.view(-1, self.nhid)
        inst = adj._indices().size()[1]

        if select_id is None:
            print("Dont have to select id.")
            row = adj._indices()[0, :]
            col = adj._indices()[1, :]
        else:
            row = adj._indices()[0, select_id].tolist()
            col = adj._indices()[1, select_id].tolist()
        patch_label = (label[row] == label[col]).long()
        pred = self.classifier(torch.cat((feature[row], feature[col]), 1))

        pos_idx = torch.nonzero(patch_label).squeeze(1).tolist()
        neg_idx = torch.nonzero(patch_label - 1).squeeze(1).tolist()
        pos_pred = pred[pos_idx]
        neg_pred = pred[neg_idx]
        pos_label = patch_label[pos_idx]
        neg_label = patch_label[neg_idx]
        print(len(pos_idx), len(neg_idx))

        # loss
        pos_loss = self.loss(pos_pred, pos_label)
        neg_loss = self.loss(neg_pred, neg_label)
        loss = 1 * pos_loss + 1 * neg_loss

        # loss = self.loss(pred, patch_label)

        # Acc
        pos_acc = compute_acc(pos_pred, pos_label)
        neg_acc = compute_acc(neg_pred, neg_label)
        return loss, pos_acc, neg_acc


class HEAD_test(nn.Module):
    def __init__(self, nhid):
        super(HEAD_test, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(
            nn.Linear(nhid * 2, nhid), nn.PReLU(nhid), nn.Linear(nhid, 2), nn.Softmax(dim=1)
        )

    def forward(self, feature1, feature2, no_list=False):
        if len(feature1.size()) == 1:
            pred = self.classifier(torch.cat((feature1, feature2)).unsqueeze(0))
            if pred[0][0] > pred[0][1]:
                is_same = False
            else:
                is_same = True
            return is_same
        else:
            pred = self.classifier(torch.cat((feature1, feature2), 1))
            # print(pred[0:10,:])
            if no_list:
                return pred[:, 1]
            score = list(pred[:, 1].cpu().detach().numpy())
            # is_same = (pred[:,0]<pred[:,1]).long()

        return score
