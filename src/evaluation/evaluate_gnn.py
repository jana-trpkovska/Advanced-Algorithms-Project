import torch
import pandas as pd
from pathlib import Path
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from src.models.gnn import GCNLinkPredictor

MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / f"src/models/gnn_model_v{MODEL_VERSION}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_RATIO = 1.0
THRESHOLD = 0.5

# -------------------- Load graph --------------------
def load_graph_data():
    nodes_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/nodes.csv")
    test_edges_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/test_edges.csv")

    node_id_map = {nid: idx for idx, nid in enumerate(nodes_df["Drug ID"].values)}
    test_edges_df["source_id"] = test_edges_df["source_id"].map(node_id_map)
    test_edges_df["target_id"] = test_edges_df["target_id"].map(node_id_map)

    test_edge_index = to_undirected(
        torch.tensor(test_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long)
    )

    num_nodes = len(node_id_map)
    return num_nodes, test_edge_index

# -------------------- Edge batch --------------------
def prepare_edge_batch(edge_index, num_nodes, neg_ratio, device):
    num_neg = int(edge_index.size(1) * neg_ratio)
    neg_edge_index = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg
    )
    edge_pairs = torch.cat([edge_index, neg_edge_index], dim=1).T.to(device)
    labels = torch.cat([
        torch.ones(edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(device)
    return edge_pairs, labels

# -------------------- Evaluate --------------------
def evaluate():
    num_nodes, test_edge_index = load_graph_data()
    print(f"Num nodes: {num_nodes}, Num test edges: {test_edge_index.size(1)}")

    model = GCNLinkPredictor(num_nodes=num_nodes, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        edge_pairs, labels = prepare_edge_batch(test_edge_index, num_nodes, NEG_RATIO, DEVICE)
        node_emb = model(test_edge_index.to(DEVICE))
        y_probs = model.predict_edge(node_emb, edge_pairs)

        y_true = labels.cpu().numpy()
        y_scores = y_probs.cpu().numpy()

        # Probability-based metrics
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)

        # Threshold-based metrics
        y_pred = (y_scores >= THRESHOLD).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n--- Test Results ---")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate()