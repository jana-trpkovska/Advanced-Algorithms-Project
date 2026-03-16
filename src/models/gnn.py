import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        """
        GCN for link prediction with learnable node embeddings.

        Args:
            num_nodes (int): Number of nodes in the graph
            embedding_dim (int): Dimension of the learnable node embeddings
            hidden_dim (int): Hidden dimension in GCN layers
            num_layers (int): Number of GCN layers
            dropout (float): Dropout rate
        """
        super(GCNLinkPredictor, self).__init__()

        # Learnable node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        in_dim = embedding_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index):
        """
        Forward pass to compute node embeddings.
        edge_index: tensor of shape [2, num_edges] with COO format
        """
        x = self.node_embeddings.weight  # [num_nodes, embedding_dim]

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        return x  # final node embeddings

    def predict_edge(self, node_emb, node_pair):
        """
        Compute probability of interaction between pairs of nodes
        node_emb: [num_nodes, hidden_dim] tensor from forward()
        node_pair: [num_pairs, 2] tensor with node indices
        """
        src, dst = node_pair[:, 0], node_pair[:, 1]
        src_emb = node_emb[src]
        dst_emb = node_emb[dst]

        # Dot product similarity
        score = torch.sum(src_emb * dst_emb, dim=1)
        return torch.sigmoid(score)
