import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    base_dir = Path(__file__).resolve().parents[2]
    edges_csv = base_dir / "data" / "datasets" / "gnn" / "edges.csv"
    output_dir = base_dir / "data" / "datasets" / "gnn"

    edges_df = pd.read_csv(edges_csv)

    # Shuffle edges
    edges_df = edges_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Split into train and temp (val+test)
    train_edges, temp_edges = train_test_split(
        edges_df,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED
    )

    # Split temp into val and test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # proportion in temp
    val_edges, test_edges = train_test_split(
        temp_edges,
        test_size=(1 - val_size),
        random_state=RANDOM_SEED
    )

    # Ensure all nodes in val/test exist in train
    train_nodes = set(train_edges["source_id"]).union(train_edges["target_id"])

    def filter_edges(df):
        return df[
            df["source_id"].isin(train_nodes) & df["target_id"].isin(train_nodes)
            ].reset_index(drop=True)

    val_edges = filter_edges(val_edges)
    test_edges = filter_edges(test_edges)

    # Save CSVs
    train_edges.to_csv(output_dir / "train_edges.csv", index=False)
    val_edges.to_csv(output_dir / "val_edges.csv", index=False)
    test_edges.to_csv(output_dir / "test_edges.csv", index=False)

    print("Edge splits complete:")
    print(f"Train edges: {len(train_edges)}")
    print(f"Validation edges: {len(val_edges)}")
    print(f"Test edges: {len(test_edges)}")


if __name__ == "__main__":
    main()
