import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_undirected
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.gnn import GNNLinkPredictor

MODEL_VERSION = 5
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / f"src/models/gnn_model_v{MODEL_VERSION}.pt"

RANDOM_SEED = 42
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.01
EPOCHS = 50
NEG_RATIO_TRAIN = 1.0
NEG_RATIO_VAL = 5.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)

def load_graph_data():
    nodes_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/nodes.csv")
    train_edges_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/train_edges.csv")
    val_edges_df = pd.read_csv(BASE_DIR / "data/datasets/gnn/val_edges.csv")

    node_id_map = {nid: idx for idx, nid in enumerate(nodes_df["Drug ID"].values)}
    for df in [train_edges_df, val_edges_df]:
        df["source_id"] = df["source_id"].map(node_id_map)
        df["target_id"] = df["target_id"].map(node_id_map)

    num_nodes = len(node_id_map)
    train_edge_index = torch.tensor(train_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long)
    val_edge_index = torch.tensor(val_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long)

    node_features = torch.tensor(nodes_df.drop("Drug ID", axis=1).values, dtype=torch.float)
    return num_nodes, to_undirected(train_edge_index), to_undirected(val_edge_index), node_features

def prepare_edge_batch(edge_index, node_features, num_nodes, neg_ratio, device):
    num_pos = edge_index.size(1)
    num_neg = int(num_pos * neg_ratio)
    pos_edge_pairs = edge_index.T

    with torch.no_grad():
        x = node_features.to(device)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(x_norm, x_norm.T)

    existing_edges = set(
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.size(1))
    )

    neg_edges = []
    attempts = 0
    max_attempts = num_neg * 10

    while len(neg_edges) < num_neg and attempts < max_attempts:
        attempts += 1
        src = torch.randint(0, num_nodes, (1,)).item()
        sim_scores = similarity[src]
        topk = torch.topk(sim_scores, k=20).indices.tolist()
        dst = topk[torch.randint(1, len(topk), (1,)).item()]

        if src == dst:
            continue
        if (src, dst) in existing_edges or (dst, src) in existing_edges:
            continue

        neg_edges.append((src, dst))

    neg_edge_pairs = torch.tensor(neg_edges, dtype=torch.long)
    edge_pairs = torch.cat([pos_edge_pairs, neg_edge_pairs], dim=0).to(device)
    labels = torch.cat([
        torch.ones(num_pos),
        torch.zeros(len(neg_edges))
    ]).to(device)
    return edge_pairs, labels

def train_step(model, edge_index, node_features, num_nodes, neg_ratio, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    edge_pairs, labels = prepare_edge_batch(edge_index, node_features, num_nodes, neg_ratio, DEVICE)

    node_emb = model(node_features.to(DEVICE), edge_index.to(DEVICE))
    preds = model.predict_edge(node_emb, edge_pairs)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, edge_index, node_features, num_nodes, neg_ratio):
    model.eval()
    with torch.no_grad():
        edge_pairs, labels = prepare_edge_batch(edge_index, node_features, num_nodes, neg_ratio, DEVICE)
        node_emb = model(node_features.to(DEVICE), edge_index.to(DEVICE))
        preds = model.predict_edge(node_emb, edge_pairs)
        roc_auc = roc_auc_score(labels.cpu(), torch.sigmoid(preds).cpu())
        pr_auc = average_precision_score(labels.cpu(), torch.sigmoid(preds).cpu())
    return roc_auc, pr_auc

def train():
    num_nodes, train_edge_index, val_edge_index, node_features = load_graph_data()
    print(f"Num nodes: {num_nodes}, Num train edges: {train_edge_index.size(1)}, Feature dim: {node_features.size(1)}")

    model = GNNLinkPredictor(node_features.size(1), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0

    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss = train_step(model, train_edge_index, node_features, num_nodes, NEG_RATIO_TRAIN, optimizer, criterion)
        val_roc, val_pr = evaluate(model, val_edge_index, node_features, num_nodes, NEG_RATIO_VAL)

        tqdm.write(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val ROC-AUC: {val_roc:.4f} | Val PR-AUC: {val_pr:.4f}")

        if val_roc > best_val_auc:
            best_val_auc = val_roc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Training complete. Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()