import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNNLinkPredictor(nn.Module):
    """
    GraphSAGE for node embeddings + MLP edge predictor
    """
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # MLP edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    def predict_edge(self, node_emb, edge_pairs):
        src, dst = edge_pairs[:, 0], edge_pairs[:, 1]
        src_emb = node_emb[src]
        dst_emb = node_emb[dst]
        edge_input = torch.cat([src_emb, dst_emb], dim=1)
        return self.edge_mlp(edge_input).squeeze()