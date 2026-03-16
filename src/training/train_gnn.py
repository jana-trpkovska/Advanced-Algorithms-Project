import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import negative_sampling, to_undirected
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from src.models.gnn import GCNLinkPredictor

# ----------------------------- Constants -----------------------------
MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "src" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / f"gnn_model_v{MODEL_VERSION}.pt"

RANDOM_SEED = 42
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.01
EPOCHS = 50
NEG_RATIO = 5  # Increased negative sampling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Data Loading -----------------------------
def load_graph_data():
    nodes_path = BASE_DIR / "data" / "datasets" / "gnn" / "nodes.csv"
    train_edges_path = BASE_DIR / "data" / "datasets" / "gnn" / "train_edges.csv"
    val_edges_path = BASE_DIR / "data" / "datasets" / "gnn" / "val_edges.csv"

    nodes_df = pd.read_csv(nodes_path)
    train_edges_df = pd.read_csv(train_edges_path)
    val_edges_df = pd.read_csv(val_edges_path)

    # Map original IDs to 0..N-1
    node_id_map = {nid: idx for idx, nid in enumerate(nodes_df["Drug ID"].values)}
    train_edges_df["source_id"] = train_edges_df["source_id"].map(node_id_map)
    train_edges_df["target_id"] = train_edges_df["target_id"].map(node_id_map)
    val_edges_df["source_id"] = val_edges_df["source_id"].map(node_id_map)
    val_edges_df["target_id"] = val_edges_df["target_id"].map(node_id_map)

    num_nodes = len(node_id_map)

    train_edge_index = torch.tensor(train_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long)
    train_edge_index = to_undirected(train_edge_index)

    val_edge_index = torch.tensor(val_edges_df[["source_id", "target_id"]].values.T, dtype=torch.long)
    val_edge_index = to_undirected(val_edge_index)

    return num_nodes, train_edge_index, val_edge_index

# ----------------------------- Edge Preparation -----------------------------
def prepare_edge_batch(edge_index, num_nodes, neg_ratio=1.0, device="cpu"):
    pos_edge_index = edge_index
    num_neg = int(pos_edge_index.size(1) * neg_ratio)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse"
    )

    edge_pairs = torch.cat([pos_edge_index, neg_edge_index], dim=1).T.to(device)
    labels = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(device)

    return edge_pairs, labels

# ----------------------------- Training Step -----------------------------
def train_step(model, edge_index, num_nodes, neg_ratio, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    edge_pairs, labels = prepare_edge_batch(edge_index, num_nodes, neg_ratio, DEVICE)
    node_emb = model(edge_index.to(DEVICE))
    preds = model.predict_edge(node_emb, edge_pairs)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------- Evaluation -----------------------------
def evaluate(model, edge_index, num_nodes, neg_ratio):
    model.eval()
    with torch.no_grad():
        edge_pairs, labels = prepare_edge_batch(edge_index, num_nodes, neg_ratio, DEVICE)
        node_emb = model(edge_index.to(DEVICE))
        preds = model.predict_edge(node_emb, edge_pairs)
        auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
        acc = accuracy_score((preds > 0.5).cpu().numpy(), labels.cpu().numpy())
    return auc, acc

# ----------------------------- Training Loop -----------------------------
def train():
    print(f"Training GNN model v{MODEL_VERSION}...")
    num_nodes, train_edge_index, val_edge_index = load_graph_data()
    print(f"Num nodes: {num_nodes}, Num train edges: {train_edge_index.size(1)}")

    model = GCNLinkPredictor(
        num_nodes=num_nodes,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_val_auc = 0.0

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Epochs"):
        loss = train_step(model, train_edge_index, num_nodes, NEG_RATIO, optimizer, criterion)
        val_auc, val_acc = evaluate(model, val_edge_index, num_nodes, NEG_RATIO)

        tqdm.write(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            tqdm.write(f"Saved best model at epoch {epoch} with Val AUC: {best_val_auc:.4f}")

    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")
    print(f"Final Val AUC: {best_val_auc:.4f}")

# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    train()
