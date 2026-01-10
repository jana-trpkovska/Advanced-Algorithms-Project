import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NEGATIVE_RATIO = 1.0  # same number of negatives and positives


def build_text(drug_name, generic_name, interacts_with):
    if pd.isna(generic_name) or generic_name.strip() == "":
        return f"{drug_name} interacts with {interacts_with}"
    return f"{drug_name} ({generic_name}) interacts with {interacts_with}"


def main():
    random.seed(RANDOM_SEED)

    base_dir = Path(__file__).resolve().parents[2]

    input_csv = base_dir / "data" / "processed" / "drug_to_drug_interactions_enriched.csv"

    output_dir = base_dir / "data" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Positive samples
    positive_samples = []

    for _, row in df.iterrows():
        text = build_text(
            drug_name=row["Drug Name"],
            generic_name=row["Generic Name"],
            interacts_with=row["Interacts With Generic Name"],
        )
        positive_samples.append((text, 1))

    positive_df = pd.DataFrame(
        positive_samples, columns=["text", "label"]
    )

    # Negative samples
    drugs = (
        df[["Drug ID", "Drug Name", "Generic Name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    positive_pairs = set(zip(df["Drug ID"], df["Interacts With Generic Name"], ))

    num_negatives = int(len(positive_df) * NEGATIVE_RATIO)
    negative_samples = set()

    while len(negative_samples) <= num_negatives:
        drug_a = drugs.sample(1).iloc[0]
        drug_b = drugs.sample(1).iloc[0]

        # Avoid self-pairing
        if drug_a["Drug ID"] == drug_b["Drug ID"]:
            continue

        # Avoid known interactions
        pair_key = (drug_a["Drug ID"], drug_b["Generic Name"])
        if pair_key in positive_pairs:
            continue

        text = build_text(
            drug_name=drug_a["Drug Name"],
            generic_name=drug_a["Generic Name"],
            interacts_with=drug_b["Generic Name"],
        )

        negative_samples.add((text, 0))

    negative_df = pd.DataFrame(list(negative_samples), columns=["text", "label"])

    # Combine and shuffle
    full_df = pd.concat([positive_df, negative_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Split into train, val, test
    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=full_df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Dataset creation complete:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()
