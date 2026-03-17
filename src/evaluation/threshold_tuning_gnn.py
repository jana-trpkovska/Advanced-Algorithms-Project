import torch
import pandas as pd
from pathlib import Path
from torch_geometric.utils import to_undirected
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from src.models.gnn import GNNLinkPredictor

MODEL_VERSION = 7
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "src" / "models" / f"gnn_model_v{MODEL_VERSION}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
NEG_RATIO = 1.0

def load_graph_data():

    nodes_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/nodes.csv")
    val_edges_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/val_edges.csv")

    node_id_map = {nid: idx for idx, nid in enumerate(nodes_df["Drug ID"].values)}

    val_edges_df["source_id"] = val_edges_df["source_id"].map(node_id_map)
    val_edges_df["target_id"] = val_edges_df["target_id"].map(node_id_map)

    val_edge_index = torch.tensor(
        val_edges_df[["source_id", "target_id"]].values.T,
        dtype=torch.long
    )

    val_edge_index = to_undirected(val_edge_index)

    feature_columns = nodes_df.columns.drop("Drug ID")

    node_features = torch.tensor(
        nodes_df[feature_columns].values,
        dtype=torch.float
    )

    return len(node_id_map), val_edge_index, node_features


def prepare_edge_batch(edge_index, node_features, num_nodes, neg_ratio, device, hard_ratio=0.7, topk=20):
    num_pos = edge_index.size(1)
    num_neg = int(num_pos * neg_ratio)
    num_hard = int(num_neg * hard_ratio)
    num_random = num_neg - num_hard
    pos_edge_pairs = edge_index.T

    with torch.no_grad():
        x = node_features.to(device)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(x_norm, x_norm.T)

    existing_edges = set(
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.size(1))
    )

    neg_edges = set()
    attempts = 0
    max_attempts = num_hard * 10

    while len(neg_edges) < num_hard and attempts < max_attempts:
        attempts += 1
        src = torch.randint(0, num_nodes, (1,)).item()
        sim_scores = similarity[src]
        topk_nodes = torch.topk(sim_scores, k=topk).indices.tolist()
        dst = topk_nodes[torch.randint(1, len(topk_nodes), (1,)).item()]

        if src == dst:
            continue
        if (src, dst) in existing_edges or (dst, src) in existing_edges:
            continue

        neg_edges.add((src, dst))

    attempts = 0
    max_attempts = num_random * 10

    while len(neg_edges) < num_neg and attempts < max_attempts:
        attempts += 1
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()

        if src == dst:
            continue
        if (src, dst) in existing_edges or (dst, src) in existing_edges:
            continue

        neg_edges.add((src, dst))

    neg_edges = list(neg_edges)

    if len(neg_edges) < num_neg:
        print(f"Warning: Only generated {len(neg_edges)} negatives out of {num_neg}")

    neg_edge_pairs = torch.tensor(neg_edges, dtype=torch.long)
    edge_pairs = torch.cat([pos_edge_pairs, neg_edge_pairs], dim=0).to(device)
    labels = torch.cat([
        torch.ones(num_pos),
        torch.zeros(len(neg_edge_pairs))
    ]).to(device)
    return edge_pairs, labels


def tune_threshold(num_steps=500):

    num_nodes, val_edge_index, node_features = load_graph_data()

    print(
        f"Num nodes: {num_nodes}, "
        f"Num val edges: {val_edge_index.size(1)}, "
        f"Feature dim: {node_features.size(1)}"
    )

    model = GNNLinkPredictor(
        in_dim=node_features.size(1),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():

        edge_pairs, labels = prepare_edge_batch(
            val_edge_index,
            node_features,
            num_nodes,
            NEG_RATIO,
            DEVICE
        )

        node_emb = model(node_features.to(DEVICE), val_edge_index.to(DEVICE))

        y_probs = model.predict_edge(node_emb, edge_pairs)

        y_true = labels.cpu().numpy()
        y_scores = y_probs.cpu().numpy()

        best_threshold = 0.5
        best_f1 = 0

        best_precision = 0
        best_recall = 0

        thresholds = np.linspace(0.3, 0.71, num_steps)

        for t in thresholds:

            y_pred = (y_scores >= t).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_precision = precision
                best_recall = recall

    print("\n--- Best Threshold ---")
    print(f"Threshold : {best_threshold:.4f}")
    print(f"Precision : {best_precision:.4f}")
    print(f"Recall    : {best_recall:.4f}")
    print(f"F1 Score  : {best_f1:.4f}")

    return best_threshold


if __name__ == "__main__":
    tune_threshold()