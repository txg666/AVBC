import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, GATv2Conv, RGCNConv, SAGEConv,
                                global_mean_pool)


class StandardGNNLayer(nn.Module):
    def __init__(self, conv, hidden_dim, dropout):
        super().__init__()
        self.conv = conv
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        if isinstance(self.conv, RGCNConv):
            x = self.conv(x, edge_index, edge_type)
        else:
            x = self.conv(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNModel(nn.Module):
    def __init__(
        self,
        gnn_type='gcn',
        in_dim=128,
        hidden_dim=256,
        out_dim=2,
        num_layers=2,
        dropout=0.5,
        num_relations=4,
        num_heads=4,
    ):
        super().__init__()

        self.gnn_type = gnn_type

        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = self.build_conv(
                gnn_type, hidden_dim, hidden_dim, num_relations, num_heads
            )
            self.layers.append(StandardGNNLayer(conv, hidden_dim, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def build_conv(self, gnn_type, in_dim, out_dim, num_relations, num_heads):
        if gnn_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif gnn_type == 'graphsage':
            return SAGEConv(in_dim, out_dim)
        elif gnn_type == 'gat':
            return GATv2Conv(in_dim, out_dim, heads=num_heads, concat=False)
        elif gnn_type == 'rgcn':
            return RGCNConv(in_dim, out_dim, num_relations)
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")

    def forward(self, data, return_embedding=False):
        x = data.x
        edge_index = data.edge_index
        edge_type = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        x = self.input_lin(x)
        x = F.relu(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)

        if batch is not None:
            x = global_mean_pool(x, batch)
        embedding = x

        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits
