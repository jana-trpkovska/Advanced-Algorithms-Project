import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[2]

    input_csv = base_dir / "data" / "processed" / "drug_to_drug_interactions_enriched.csv"
    output_dir = base_dir / "data" / "datasets" / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Build node list
    nodes = df[["Drug ID", "Drug Name", "Generic Name"]].drop_duplicates().reset_index(drop=True)
    nodes.to_csv(output_dir / "nodes.csv", index=False)

    # Create mapping: generic_name -> node_id
    generic_to_id = dict(zip(nodes["Generic Name"], nodes["Drug ID"]))

    # Build edges
    edges = []
    for _, row in df.iterrows():
        source_id = generic_to_id[row["Generic Name"]]
        target_generic = row["Interacts With Generic Name"]
        if target_generic not in generic_to_id:
            continue  # skip if target not in nodes (unlikely)
        target_id = generic_to_id[target_generic]
        if source_id == target_id:
            continue  # skip self-loop
        edges.append((source_id, target_id))

    # Remove duplicate edges
    edges = list(set(edges))

    edges_df = pd.DataFrame(edges, columns=["source_id", "target_id"])
    edges_df.to_csv(output_dir / "edges.csv", index=False)

    print(f"Nodes: {len(nodes)}, Edges: {len(edges_df)}")
    print("GNN dataset creation complete.")

if __name__ == "__main__":
    main()
