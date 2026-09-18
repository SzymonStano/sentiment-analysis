import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import yaml


def main() -> None:
    # Load parameters from YAML config file
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset_name = args.dataset

    split_params = params["split"]
    dataset_cfg = params["datasets"][dataset_name]

    # Load preprocessed dataset
    input_path = f"/app/data/preprocessed/{dataset_name}/preprocessed.csv"
    data = pd.read_csv(input_path, engine="python")
    class_column = dataset_cfg["label_column"]

    # Split features and target
    X = data.drop(class_column, axis=1)
    y = data[class_column]

    # Enable stratified split if configured
    stratify = y if split_params.get("stratify", None) == "y" else None

    # perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_params["test_size"], stratify=stratify
    )

    # Concatenate features and labels for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save the split datasets to CSV files
    train_data.to_csv(
        f"/app/data/train_test/{dataset_name}_train_data.csv", index=False
    )
    test_data.to_csv(f"/app/data/train_test/{dataset_name}_test_data.csv", index=False)


if __name__ == "__main__":
    main()
