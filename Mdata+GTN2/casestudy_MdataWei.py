# -*- coding:utf-8 -*-
import argparse
import copy
# 这个是复制了train3main，在这个基础上画图，其他和3一样

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from sklearn.metrics import precision_score
from torch import nn
from torch_geometric.utils import add_self_loops, negative_sampling
from tqdm import tqdm
import torch.nn.functional as F

# from model import PygFastGTNs
from model_ori import FastGTNs
from new_model import GraphSAGE, GAT, GCN
from utils import _norm, init_seed

import time
import os
import os.path as osp
from methods import Accuracy_Precision_Sensitivity_Specificity_MCC, average_list, sum_list
from plt_log import draw_log

device = torch.device('cuda')
# device = torch.device('cpu')

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse

"原始版本（也就是详细版本在D:\Python_project\Jupyternotebook\Siri_DataProcessing\CTD\数据处理\-数据预处理的Untitled"
""
""

# 读取次数
left = pd.read_csv('Num_left_fre0829.csv', encoding='gbk')
right = pd.read_csv('Num_right_fre0829.csv', encoding='gbk')

left_values = left.values
right_values = right.values
# count = np.concatenate([left_values, right_values], axis=0)
count = left_values
all_positive_nodes_idx = count[:, 0].tolist()
#
count_dict = dict(zip(count[:, 0], count[:, 1]))

sdf = pd.read_csv("../Data_sr2/5_C_D.csv")
node_list = pd.read_csv("../Data_sr2/node_list0829.csv")
n_nodes = node_list['Node'].unique()  # 获得所有产品的id

# 3、生成A_CD稀疏矩阵
CIDs = []
DIDs = []
for i, values in enumerate(sdf.values):
    CIDs.append(values[0])
    DIDs.append(values[1])

name2features = {}
for i, values in enumerate(node_list.values):
    idx = node_list.index[i]
    name = values[0]
    name2features[name] = idx

row_indexes = []
for cid in CIDs:
    idx = name2features[cid]
    row_indexes.append(idx)

col_indexes = []
for did in DIDs:
    idx = name2features[did]
    col_indexes.append(idx)

data = np.ones_like(col_indexes)
node_num = len(n_nodes)
A_CD = csr_matrix((data, (row_indexes, col_indexes)), shape=(node_num, node_num))
A_DC = A_CD.transpose()

# 4、生成A_CG稀疏矩阵
sdf2 = pd.read_csv("../Data_sr2/6_C_G.csv")
CIDs2 = []
GIDs = []
for i, values in enumerate(sdf2.values):
    CIDs2.append(values[0])
    GIDs.append(values[1])

row_indexes2 = []
for cid in CIDs2:
    idx = name2features[cid]
    row_indexes2.append(idx)

col_indexes2 = []
for gid in GIDs:
    idx = name2features[gid]
    col_indexes2.append(idx)

data2 = np.ones_like(col_indexes2)
A_CG = csr_matrix((data2, (row_indexes2, col_indexes2)), shape=(node_num, node_num))
A_GC = A_CG.transpose()

# 5、生成A_GD稀疏矩阵
sdf3 = pd.read_csv("../Data_sr2/6_G_D.csv")
GIDs2 = []
DIDs2 = []
score = []
for i, values in enumerate(sdf3.values):
    GIDs2.append(values[0])
    DIDs2.append(values[1])
    score.append(values[2])

row_indexes3 = []
for gid in GIDs2:
    idx = name2features[gid]
    row_indexes3.append(idx)

col_indexes3 = []
for did in DIDs2:
    idx = name2features[did]
    col_indexes3.append(idx)

data3 = np.array(score)
# data3 = np.ones_like(col_indexes3)

A_GD = csr_matrix((data3, (row_indexes3, col_indexes3)), shape=(node_num, node_num))
A_DG = A_GD.transpose()
edges = [A_CG, A_GC, A_GD, A_DG]

# 5、Node Feasures
fea1 = pd.read_csv("../Data_sr2/drug_feature0829.csv", index_col=0)
matrix1 = fea1.values

fea2 = pd.read_csv("../Data_sr2/gene_feature0829.csv", index_col=0)
fea2.fillna(0, inplace=True)
matrix2 = fea2.values

fea5 = pd.read_csv("../Data_sr2/disease_feature0829.csv", index_col=0)
fea5.fillna(0, inplace=True)
matrix5 = fea5.values

node_faeture = np.concatenate((matrix1, matrix2, matrix5))

# 正边索引

pos_edges = [A_CD, A_DC]

edge_index = []

for i, edge in enumerate(pos_edges):
    edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
    # normalize each adjacency matrix
    # edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp,fill_value=1e-20, num_nodes=num_nodes)
    # deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
    # value_tmp = deg_inv_sqrt[deg_row] * value_tmp
    edge_index.append((edge_tmp, value_tmp))

df = pd.read_csv("../Data_sr2/final_weight09061.csv")
row = df.iloc[:,0]
col = df.iloc[:,1]
sc = df.iloc[:,2]


def split_dataset(adj):
    # adj p-a a-p p-s s-p
    # edge_index1 = graph['paper', 'to', 'author'].edge_index
    # edge_index2 = graph['author', 'to', 'paper'].edge_index

    edge_index1 = adj[1][0]
    # edge_index2 = adj[1][0]
    # edge_indx = torch.cat((edge_index1, edge_index2), dim=1)
    edge_indx = edge_index1
    pos_edge_label_index = copy.deepcopy(edge_indx)
    pos_edge_label = torch.ones(pos_edge_label_index.shape[1])

    # 复边读进来
    negativeSample = pd.read_csv("../Data_sr2/NegativeSample0829.csv", header=None)
    neg_edge_index = torch.tensor(negativeSample.values, dtype=torch.long).T.to(device)
    neg_edge_label = torch.zeros(neg_edge_index.shape[1])

    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)

    edge_label = edge_label.to(device)
    edge_label_index = edge_label_index.to(device)

    print(edge_label.shape)
    print(edge_label_index.shape)

    # total_num_edges = edge_label_index.shape[1]
    # # 打乱再取811
    # num_train = int(total_num_edges * 0.8)
    # step_length = int(total_num_edges * 0.2)

    index_shuffled = torch.randperm(edge_label_index.size(1))
    index = index_shuffled.cpu().numpy().tolist()
    edge_label = edge_label[index_shuffled]
    edge_label_index = edge_label_index[:, index_shuffled]

    return edge_label, edge_label_index


class PygGTNs_LP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = FastGTNs(num_edge_type=num_edge_type,
                                w_in=node_features.shape[1],
                                num_nodes=num_nodes,
                                num_class=3,
                                args=args)

        # self.att = nn.Linear(2 * args.node_dim, 1)
        # self.encoder = GCN(node_features.shape[1], 64, 256)
        # self.w = nn.Linear(2 * args.node_dim, 2)

        # self.w1 = nn.Linear(256, 128)
        # self.w2 = nn.Linear(128, 64)
        # self.w3 = nn.Linear(64, 2)

        # # 输出的概率不同，但药的类别一样
        # self.w1 = nn.Linear(512, 128)
        # self.w2 = nn.Linear(128, 2)
        # self.bn = nn.BatchNorm1d(512)

        # self.w = nn.Sequential(
        #     # nn.BatchNorm1d(512),
        #     nn.Linear(args.node_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 2),
        # )

        self.w = nn.Sequential(
            nn.BatchNorm1d(2 * args.node_dim),
            nn.Linear(2 * args.node_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )
        self.input_norm = nn.LayerNorm(node_features.shape[1])


    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.01)

    def encode(self):
        x = self.input_norm(node_features)
        x = self.encoder(A, x, num_nodes=num_nodes)
        # x = self.encoder(node_features, A)
        return x

    def decode(self, z, edge_label_index):
        # z
        # print(z.shape, edge_label_index.shape)
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())
        # r = (src * dst).sum(dim=-1)
        # print(r.size())

        # # cat前加入一点小小的attention
        # a1 = self.att(torch.cat((src, dst), dim=-1))
        # a2 = self.att(torch.cat((dst, src), dim=-1))
        # c = a1 * src + a2 * dst
        c = torch.cat((src, dst), dim=-1)

        r = self.w(c)
        # print(r.shape)  # 31836 2

        return r

    def forward(self, edge_label_index):
        z = self.encode()
        out = self.decode(z, edge_label_index)
        # Softmax = nn.Softmax(dim=1)
        # out = Softmax(out)
        return out

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def test(args, model, edge_label_index, j):
    SCORE = []
    PRED = []
    model.eval()
    with torch.no_grad():
        # z = model.encode()
        # out = model.decode(z, edge_label_index)

        out = model(edge_label_index)

        _, pred = out.max(dim=1)
        # sco_1, pred = out.max(dim=1)
        sco = F.softmax(out, dim=1)
        scores = sco[:, 1]
        SCORE.extend(scores)
        # LABEL.extend(edge_label)
        PRED.extend(pred)
        # if (epoch + 1) == args.epoch:
        #     SCORE.extend(scores)
        #     # LABEL.extend(edge_label)
        #     PRED.extend(pred)
        # model.train()
    SCO = [x.item() for x in SCORE]
    SCORE.clear()
    a = np.array(SCO)
    SCO.clear()
    # a = np.array(SCORE)
    ss = pd.DataFrame(a.T)
    output1 = 'Siridataset/score' + str(j) + '.csv'
    ss.to_csv(output1)

    PRE = [x.item() for x in PRED]
    PRED.clear()
    P = np.array(PRE)
    PRE.clear()
    PP = pd.DataFrame(P.T)
    output2 = 'Siridataset/pred_label' + str(j) + '.csv'
    PP.to_csv(output2)
    print('完成！')

    # test_roc, test_f1, test_aupr = roc_auc_score(edge_label.cpu().numpy(), scores.cpu().numpy()), f1_score(edge_label.cpu().numpy(), pred.cpu().numpy()), average_precision_score(edge_label.cpu().numpy(),scores.cpu().numpy())
    # return test_roc, test_f1, test_aupr, val_loss

    # return roc_auc_score(edge_label.cpu().numpy(), scores.cpu().numpy()), f1_score(edge_label.cpu().numpy(), pred.cpu().numpy()), average_precision_score(edge_label.cpu().numpy(), scores.cpu().numpy())


def train(args, train_index, train_label):
    epoch_list = []
    lossTr_list = []
    # lossVal_list = []

    model = PygGTNs_LP(args).to(device)
    model.init()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = torch.FloatTensor([0.5, 0.5])
    criterion = nn.CrossEntropyLoss(weight, reduction='none').to(device)
    model.train()
    for epoch in tqdm(range(args.epoch)):
        optimizer.zero_grad()
        out = model(train_index)
        # softmax_func = nn.Softmax(dim=1)
        # softmax_out = softmax_func(out)
        # log_output = torch.log(softmax_out)
        # nll_loss_func = nn.NLLLoss()
        # loss = nll_loss_func(log_output, train_label.long())

        loss = criterion(out, train_label.long())
        loss = (loss * sample_weight.to(device)).sum()
        # test(args, model, test_edge_label_index, epoch)

        # test
        # train_acc = test(model, train_label, train_index)
        # TP, FP, FN, TN = Accuracy_Precision_Sensitivity_Specificity_MCC(args, model, test_label, test_index, device)
        # test_roc, test_f1, test_aupr, val_loss = test(args, model, test_edge_label, test_edge_label_index, epoch)

        # ---------------画图--------------------------------
        epoch_list.append(epoch)
        lossTr_list.append(loss.cpu().detach().numpy())
        # lossVal_list.append(val_loss.cpu().detach().numpy())
        # ---------------画图--------------------------------
        loss.backward()
        optimizer.step()
        # print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} test_recall {:.4f} test_roc {:.4f}'
        #       .format(epoch, loss.item(), val_loss.item(), test_aupr, test_roc))
        print('epoch {:03d} train_loss {:.8f}'.format(epoch, loss.item()))

    # test(args, model, test_edge_label_index, epoch)
    torch.save(model, "./Siridataset/srmodel.pth")
    # return TP, FP, FN, TN, test_roc, test_f1, test_aupr,lossTr_list, lossVal_list, epoch_list
    return lossTr_list, epoch_list


if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FastGTNs',
                        help='Model')
    parser.add_argument('--dataset', type=str, default='data0212',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='mean')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0,
                        help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=3,
                        help='number of non-local negibors')
    parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
    parser.add_argument('--num_FastGTN_layers', type=int, default=1,
                        help='number of FastGTN layers')

    parser.add_argument("--trainingName", default='data0324', help="the name of this training")
    parser.add_argument("--crossValidation", type=int, default=1, help="do cross validation")
    parser.add_argument("--foldNumber", type=int, default=1, help="fold number of cross validation")

    parser.add_argument('--savedir', default="Siridataset/", help="directory to save the loss picture")

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    edges = edges
    node_features = node_faeture
    num_nodes = edges[0].shape[0]

    args.num_nodes = num_nodes
    # build adjacency matrices for each edge type
    # p-a a-p p-s s-p
    A = []
    num_edges = []
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        # print(edge_tmp)
        # if i > 1:
        #     value_tmp = torch.from_numpy(data3).type(torch.cuda.FloatTensor)
        # else:
        #     value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        num_edges.append(edge_tmp.size(1))
        # normalize each adjacency matrix
        edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp,
                                             fill_value=1e-20, num_nodes=num_nodes)
        deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
        value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp, value_tmp))

    # # 第五个邻接矩阵--不用权重
    # edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
    # value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    # A.append((edge_tmp, value_tmp))

    print('init:')
    for x, y in A:
        print(x.shape, y.shape)
    print(node_features.shape)

    # 第五个邻接矩阵----使用权重

    edge_tmp = torch.from_numpy(np.vstack((col.values, row.values))).type(torch.cuda.LongTensor)
    value_tmp = torch.from_numpy(sc.values).type(torch.cuda.FloatTensor)
    A.append((edge_tmp, value_tmp))

    print('----------')
    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    print(node_features.shape)
    print('num_nodes=', num_nodes)

    train_edge_label, train_edge_label_index = split_dataset(edge_index)
    # 根据train_edge_index计算样本权重
    sample_weight = np.ones_like(train_edge_label_index.cpu().numpy()[0]).astype(float)
    # 将正样本中以各自出现的次数
    train_idx = train_edge_label_index.cpu().numpy().T.tolist()
    idx = []
    idx_count = []
    for k in range(len(train_idx)):
        if train_idx[k][0] in all_positive_nodes_idx and count_dict[train_idx[k][0]] > 100:
            idx.append(k)
            idx_count.append(count_dict[train_idx[k][0]] / 10)
        elif train_idx[k][1] in all_positive_nodes_idx and count_dict[train_idx[k][1]] > 100:
            idx.append(k)
            idx_count.append(count_dict[train_idx[k][1]] / 10)
    idx = np.array(idx)
    # 得到每个位置应该除以的数
    print(sample_weight)
    sample_weight[idx] = sample_weight[idx] / np.array(idx_count)
    sample_weight /= np.sum(sample_weight)
    sample_weight = torch.FloatTensor(sample_weight)

    print('数据集划分完成')

    T = train(args, train_edge_label_index, train_edge_label)

    # 循环
    saving_path = f'result'
    if osp.exists(saving_path):
        print('There is already a training of the same name')
    # raise Exception('There is already a training of the same name')
    else:
        os.makedirs(saving_path)
    model = torch.load("./Siridataset/srmodel.pth")
    for i in range(0, 7):
        j = i + 1
        input = '../casestudy0829/Nega_case' + str(j) + '.csv'
        print(input)
        nega_case = pd.read_csv(input, header=None)
        test_edge_label_index = torch.tensor(nega_case.values, dtype=torch.long).T.to(device)
        test(args, model, test_edge_label_index, j)
