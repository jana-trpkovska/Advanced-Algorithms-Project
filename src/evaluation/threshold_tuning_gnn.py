import torch
import pandas as pd
from pathlib import Path
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from src.models.gnn import GCNLinkPredictor

MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / f"src/models/gnn_model_v{MODEL_VERSION}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_RATIO = 1.0
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# -------------------- Load graph --------------------
def load_graph_data():
    nodes_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/nodes.csv")
    val_edges_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/val_edges.csv")

    node_id_map = {nid: idx for idx, nid in enumerate(nodes_df["Drug ID"].values)}
    val_edges_df["source_id"] = val_edges_df["source_id"].map(node_id_map)
    val_edges_df["target_id"] = val_edges_df["target_id"].map(node_id_map)

    val_edge_index = torch.tensor(val_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long)
    val_edge_index = to_undirected(val_edge_index)

    num_nodes = len(node_id_map)
    return num_nodes, val_edge_index

# -------------------- Edge batch --------------------
def prepare_edge_batch(edge_index, num_nodes, neg_ratio, device):
    num_neg = int(edge_index.size(1) * neg_ratio)
    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg)
    edge_pairs = torch.cat([edge_index, neg_edge_index], dim=1).T.to(device)
    labels = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]).to(device)
    return edge_pairs, labels

# -------------------- Threshold tuning --------------------
def tune_threshold(num_steps=500):
    num_nodes, val_edge_index = load_graph_data()
    print(f"Num nodes: {num_nodes}, Num val edges: {val_edge_index.size(1)}")

    model = GCNLinkPredictor(num_nodes=num_nodes, embedding_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
                             num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        edge_pairs, labels = prepare_edge_batch(val_edge_index, num_nodes, NEG_RATIO, DEVICE)
        node_emb = model(val_edge_index.to(DEVICE))
        y_probs = model.predict_edge(node_emb, edge_pairs)

        y_true = labels.cpu().numpy()
        y_scores = y_probs.cpu().numpy()

        best_threshold = 0.5
        best_f1 = 0
        best_precision = 0
        best_recall = 0

        thresholds = np.linspace(0.01, 1.0, num_steps)
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