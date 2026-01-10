import pandas as pd
from pathlib import Path


def enrich_interactions_with_generic_names(
    interactions_path: Path,
    drugs_data_path: Path,
    output_path: Path
):
    interactions_df = pd.read_csv(interactions_path)
    drugs_df = pd.read_csv(drugs_data_path)

    id_to_generic = (drugs_df.set_index("Drug ID")["Generic Name"].to_dict())

    interactions_df["Generic Name"] = interactions_df["Drug ID"].map(id_to_generic)

    missing_generic = interactions_df["Generic Name"].isna().sum()
    if missing_generic > 0:
        print(f"Warning: {missing_generic} rows have missing generic names.")

    interactions_df = interactions_df[
        [
            "Drug ID",
            "Drug Name",
            "Generic Name",
            "Interacts With Generic Name",
            "Interaction"
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    interactions_df.to_csv(output_path, index=False)

    print(f"Enriched dataset saved to: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    interactions_csv = BASE_DIR / "data" / "raw" / "drug_to_drug_interactions_final.csv"
    drugs_data_csv = BASE_DIR / "data" / "raw" / "drugs_data_final.csv"
    output_csv = BASE_DIR / "data" / "processed" / "drug_to_drug_interactions_enriched.csv"

    enrich_interactions_with_generic_names(
        interactions_path=interactions_csv,
        drugs_data_path=drugs_data_csv,
        output_path=output_csv
    )
