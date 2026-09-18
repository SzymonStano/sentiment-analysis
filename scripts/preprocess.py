import yaml
import pandas as pd
import glob
import argparse
from typing import Dict, List
from sklearn.model_selection import train_test_split


def load_sephora(cfg: Dict) -> pd.DataFrame:
    """
    Loads and preprocesses the Sephora dataset.
    - Reads product and review data.
    - Merges reviews with product information.
    - Filters out rows with missing labels.
    - Optionally downsamples to a max number of samples.
    """
    print("Loading Sephora dataset")

    # Load product data
    products = pd.read_csv(cfg["product_path"], engine="python")

    # Load and concatenate all review files
    review_files = glob.glob(cfg["review_glob"])
    all_reviews = pd.concat(
        [pd.read_csv(f, engine="python") for f in review_files], ignore_index=True
    )

    # Merge reviews with product metadata
    df = pd.merge(all_reviews, products, on=cfg["merge_on"], how="left")

    # Ensure non-empty dataset and remove rows with missing labels
    assert not df.empty, "Merged Sephora dataframe is empty"
    df = df.dropna(subset=cfg["label_column"])

    # Stratified downsampling (optional)
    df, _ = train_test_split(
        df,
        train_size=cfg["max_samples"],
        stratify=df[cfg["label_column"]],
        random_state=42,
    )

    # Fix duplicate column name
    df = df.rename(columns={"price_usd_x": "price_usd"})

    assert not df.empty, "Loaded sephora dataset is empty"
    return df


def load_movies(cfg: Dict) -> pd.DataFrame:
    """
    Loads and preprocesses the Movies dataset.
    - Reads positive and negative reviews from text files.
    - Assigns binary labels (1 for positive, 0 for negative).
    """
    print("Loading Movies dataset")

    def load_file_to_series(filepath: str) -> pd.Series:
        # Reads each line from a text file into a Series
        with open(filepath, "r", encoding=cfg["encoding"]) as f:
            return pd.Series([str(line).strip() for line in f], dtype="string")

    # Load positive and negative examples
    pos_text = load_file_to_series(cfg["pos_path"])
    neg_text = load_file_to_series(cfg["neg_path"])

    # Combine into a single DataFrame with labels
    df = pd.DataFrame(
        {
            "text": pd.concat([pos_text, neg_text], ignore_index=True),
            "class": pd.Series([1] * len(pos_text) + [0] * len(neg_text), dtype="int8"),
        }
    )

    # Ensure correct type and non-empty dataset
    df["text"] = df["text"].astype("string")
    assert not df.empty, "Loaded movies dataset is empty"
    return df


def main() -> None:
    # Load global configuration from YAML
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    # Parse CLI argument for dataset name
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset_name = args.dataset

    cfg = config["datasets"][dataset_name]

    drop_columns = cfg.get("drop_columns", [])
    class_column = cfg.get("label_column", [])

    # Load the selected dataset
    if dataset_name == "sephora":
        df = load_sephora(cfg)
    elif dataset_name == "movies":
        df = load_movies(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Drop unnecessary columns and rows with missing labels
    df = df.drop(columns=drop_columns)
    df = df.dropna(subset=class_column)

    # Save the cleaned and preprocessed dataset
    df.to_csv(f"/app/data/preprocessed/{dataset_name}/preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
