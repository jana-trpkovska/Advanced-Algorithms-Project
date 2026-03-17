import torch
import pandas as pd
from pathlib import Path
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from src.models.gnn import GNNLinkPredictor

MODEL_VERSION = 7
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / f"src/models/gnn_model_v{MODEL_VERSION}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_RATIO = 1.0
THRESHOLD = 0.3715 # after threshold tuning

def load_graph_data():
    nodes_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/nodes.csv")
    test_edges_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/test_edges.csv")

    node_id_map = {nid: idx for idx, nid in enumerate(nodes_df["Drug ID"].values)}
    test_edges_df["source_id"] = test_edges_df["source_id"].map(node_id_map)
    test_edges_df["target_id"] = test_edges_df["target_id"].map(node_id_map)

    test_edge_index = to_undirected(torch.tensor(test_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long))
    node_features = torch.tensor(nodes_df.drop("Drug ID", axis=1).values, dtype=torch.float)
    return len(node_id_map), test_edge_index, node_features

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

def evaluate():
    num_nodes, test_edge_index, node_features = load_graph_data()
    print(f"Num nodes: {num_nodes}, Num test edges: {test_edge_index.size(1)}, Feature dim: {node_features.size(1)}")

    model = GNNLinkPredictor(node_features.size(1), hidden_dim=128, num_layers=2, dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        edge_pairs, labels = prepare_edge_batch(test_edge_index, node_features, num_nodes, NEG_RATIO, DEVICE)
        node_emb = model(node_features.to(DEVICE), test_edge_index.to(DEVICE))
        logits = model.predict_edge(node_emb, edge_pairs)
        probs = torch.sigmoid(logits)

        y_true = labels.cpu().numpy()
        y_scores = probs.cpu().numpy()
        y_pred = (y_scores >= THRESHOLD).astype(int)

        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    print("\n--- Test Results ---")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate()