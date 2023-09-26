import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)
        self.norm = nn.BatchNorm1d(h_feats)
        self.input_fc = nn.Linear(in_feats, h_feats)
        self.output_fc = nn.Linear(h_feats, out_feats)

    def forward(self, x, adj):
        # 拼接
        edge_index, edge_weight = adj[0][0], adj[0][1]
        for k in range(1, len(adj)):
            edge_index = torch.cat((edge_index, adj[k][0]), dim=-1)
            edge_weight = torch.cat((edge_weight, adj[k][1]), dim=-1)

        # x = self.conv1(x, edge_index, edge_weight) + self.input_fc(x)
        # x = self.norm(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.5, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight) + self.output_fc(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, out_feats)
        self.norm = nn.BatchNorm1d(h_feats)
        self.input_fc = nn.Linear(in_feats, h_feats)
        self.output_fc = nn.Linear(h_feats, out_feats)

    def forward(self, x, adj):
        # 拼接
        edge_index, edge_weight = adj[0][0], adj[0][1]
        for k in range(1, len(adj)):
            edge_index = torch.cat((edge_index, adj[k][0]), dim=-1)
            edge_weight = torch.cat((edge_weight, adj[k][1]), dim=-1)

        # x = self.conv1(x, edge_index) + self.input_fc(x)
        # x = self.norm(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.5, training=self.training)
        # x = self.conv2(x, edge_index) + self.output_fc(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=4, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=4, concat=False)

    def forward(self, x, adj):
        # 拼接
        edge_index, edge_weight = adj[0][0], adj[0][1]
        for k in range(1, len(adj)):
            edge_index = torch.cat((edge_index, adj[k][0]), dim=-1)
            edge_weight = torch.cat((edge_weight, adj[k][1]), dim=-1)

        # x = torch.FloatTensor(node_features)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x
