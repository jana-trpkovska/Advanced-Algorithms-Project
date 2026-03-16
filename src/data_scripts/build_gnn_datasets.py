from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def main():
    base_dir = Path(__file__).resolve().parents[2]
    input_edges_csv = base_dir / "data" / "processed" / "drug_to_drug_interactions_enriched.csv"
    input_drugs_csv = base_dir / "data" / "raw" / "drugs_data_final.csv"
    output_dir = base_dir / "data" / "datasets" / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_df = pd.read_csv(input_edges_csv)
    drugs_df = pd.read_csv(input_drugs_csv)

    nodes = drugs_df[["Drug ID", "Drug Name", "Generic Name", "Drug Class", "Usage", "Warnings",
                      "Side Effects"]].drop_duplicates().reset_index(drop=True)

    class_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    class_features = class_encoder.fit_transform(nodes[["Drug Class"]])
    class_feature_cols = [f"class_{c}" for c in class_encoder.categories_[0]]
    class_features_df = pd.DataFrame(class_features, columns=class_feature_cols)

    text_columns = ["Usage", "Warnings", "Side Effects"]
    nodes["combined_text"] = nodes[text_columns].fillna("").agg(" ".join, axis=1)

    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    text_features = tfidf_vectorizer.fit_transform(nodes["combined_text"]).toarray()
    text_feature_cols = [f"text_{i}" for i in range(text_features.shape[1])]
    text_features_df = pd.DataFrame(text_features, columns=text_feature_cols)

    node_features_df = pd.concat([nodes[["Drug ID"]], class_features_df, text_features_df], axis=1)

    node_features_df.to_csv(output_dir / "nodes.csv", index=False)
    print(
        f"Nodes saved with features: {len(node_features_df)} nodes, feature dimension: {class_features_df.shape[1] + text_features_df.shape[1]}")

    generic_to_id = dict(zip(nodes["Generic Name"], nodes["Drug ID"]))
    edges = []
    for _, row in edges_df.iterrows():
        source_id = generic_to_id.get(row["Generic Name"])
        target_id = generic_to_id.get(row["Interacts With Generic Name"])
        if source_id is None or target_id is None or source_id == target_id:
            continue
        edges.append((source_id, target_id))
    edges = list(set(edges))
    edges_df_final = pd.DataFrame(edges, columns=["source_id", "target_id"])
    edges_df_final.to_csv(output_dir / "edges.csv", index=False)

    print(f"Edges saved: {len(edges_df_final)} edges")
    print("GNN dataset creation complete.")


if __name__ == "__main__":
    main()