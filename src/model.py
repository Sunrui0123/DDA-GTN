import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from gcn import GCNConv
import torch_sparse
from torch_geometric.utils import softmax
from utils import _norm, generate_non_local_graph

device = f'cuda' if torch.cuda.is_available() else 'cpu'


class FastGTNs(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
        super(FastGTNs, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        fastGTNs = []
        # print('num=', num_edge_type)
        fastGTNs.append(FastGTN(num_edge_type, w_in, num_class, num_nodes, args))
        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.linear = nn.Linear(args.node_dim, num_class)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, A, X, target_x, target, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None,
                epoch=None):
        if num_nodes == None:
            num_nodes = self.num_nodes
        # print(num_nodes)   # 8994
        # print(X.shape)   # 8994 1902
        # print(target_x.shape)   # 600
        # print(target.shape)  # 600 3
        H_, Ws = self.fastGTNs[0](A, X, num_nodes=num_nodes, epoch=epoch)
        y = self.linear(H_[target_x])
        loss = self.loss(y, target.squeeze())
        # print(y.shape, target.squeeze().shape, H_.shape)
        return loss, y, Ws


class FastGTN(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None, pre_trained=None):
        super(FastGTN, self).__init__()
        if args.non_local:
            num_edge_type += 1
        self.num_edge_type = num_edge_type
        self.num_channels = args.num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        args.w_in = w_in
        self.w_out = args.node_dim
        self.num_class = num_class
        self.num_layers = args.num_layers

        if pre_trained is None:
            layers = []
            layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args))
            self.layers = nn.ModuleList(layers)

        self.Ws = []
        for i in range(self.num_channels):
            self.Ws.append(GCNConv(in_channels=self.w_in, out_channels=self.w_out).weight)
        self.Ws = nn.ParameterList(self.Ws)

        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)

        feat_trans_layers = []
        for i in range(self.num_layers + 1):
            feat_trans_layers.append(nn.Sequential(nn.Linear(self.w_out, 128),
                                                   nn.ReLU(),
                                                   nn.Linear(128, 64)))
        self.feat_trans_layers = nn.ModuleList(feat_trans_layers)

        self.args = args

        self.out_norm = nn.LayerNorm(self.w_out)
        self.relu = torch.nn.ReLU()

    def forward(self, A, X, num_nodes, eval=False, node_labels=None, epoch=None):
        #
        # print(A[0])
        # print(X.shape)   # 8994 1902
        # print(num_nodes)   # 8994
        Ws = []
        X_ = [X @ W.to(X.dtype) for W in self.Ws]   # GCN
        H = [X @ W.to(X.dtype) for W in self.Ws]   # GCN

        for i in range(self.num_layers):
            H, W = self.layers[i](H, A, num_nodes, epoch=epoch, layer=i + 1)
            Ws.append(W)   # 经过FastLayer

        for i in range(self.num_channels):
            if i == 0:
                H_ = F.relu(self.args.beta * (X_[i]) + (1 - self.args.beta) * H[i])
            else:
                H_ = torch.cat((H_, F.relu(self.args.beta * (X_[i]) + (1 - self.args.beta) * H[i])), dim=1)

        H_ = F.relu(self.linear1(H_))

        return H_, Ws


class FastGTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, first=True, args=None, pre_trained=None):
        super(FastGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args)
        self.args = args
        self.feat_transfrom = nn.Sequential(nn.Linear(args.w_in, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64))


    def forward(self, H_, A, num_nodes, epoch=None, layer=None):
        result_A, W1 = self.conv1(A, num_nodes, epoch=epoch, layer=layer)
        W = [W1]
        Hs = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)
            H = torch.sparse.mm(mat_a, H_[i])
            Hs.append(H)
        return Hs, W


class FastGTConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, args=None, pre_trained=None):
        super(FastGTConv, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.1)

    def forward(self, A, num_nodes, epoch=None, layer=None):
        weight = self.weight
        filter = F.softmax(weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value * filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value * filter[i][j]))

            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes,
                                                 op='add')
            results.append((index, value))

        return results, filter


class PygFastGTNs(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
        super(PygFastGTNs, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        fastGTNs = []

        fastGTNs.append(FastGTN(num_edge_type, w_in, num_class, num_nodes, args))
        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.linear = nn.Linear(args.node_dim, num_class)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, A, X, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None,
                epoch=None):
        if num_nodes == None:
            num_nodes = self.num_nodes
        # print(num_nodes)   # 8994
        # print(X.shape)   # 8994 1902
        # print(target_x.shape)   # 600
        # print(target.shape)  # 600 3
        H_, Ws = self.fastGTNs[0](A, X, num_nodes=num_nodes, epoch=epoch)
        return H_
