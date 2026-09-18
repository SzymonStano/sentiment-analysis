import pandas as pd
import yaml
import json
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.pipelines import build_sephora_pipeline, build_movies_pipeline
from typing import Dict, Any


def load_best_params(model_name: str, dataset_name: str) -> Dict[str, Any]:
    """
    Loads best hyperparameters from JSON file and strips pipeline step prefixes.
    """
    filename = f"results_hp_tuning/{dataset_name}_{model_name}_tuning.json"
    with open(filename) as f:
        best_result = json.load(f)
    params = best_result["best_params"]
    # Remove step names (e.g., 'clf__n_estimators' -> 'n_estimators')
    params_extracted = {k.split("__")[1]: v for k, v in params.items()}
    return params_extracted


def main() -> None:
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["rf", "svm", "dummy"])
    args = parser.parse_args()

    # Load general parameters from YAML file
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dataset_name = args.dataset
    model_name = args.model
    dataset_cfg = params["datasets"][dataset_name]
    class_column = dataset_cfg["label_column"]
    feature_engineering = params["train"]["feature_engineering"]
    vectorizer = params["train"]["vectorizer"]

    # load data
    df_train = pd.read_csv(f"data/train_test/{dataset_name}_train_data_prepared.csv")
    df_test = pd.read_csv(f"data/train_test/{dataset_name}_test_data_prepared.csv")

    y_train = df_train[class_column]
    y_test = df_test[class_column]

    # Select and initialize model
    if model_name == "rf":
        model_params = load_best_params(model_name, dataset_name)
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_name == "svm":
        model_params = load_best_params(model_name, dataset_name)
        model = SVC(**model_params, random_state=42)
    elif model_name == "dummy":
        model_params = {"strategy": "most_frequent"}
        model = DummyClassifier(**model_params, random_state=42)
    else:
        raise ValueError("Nieobsługiwany model")

    # Select and build appropriate pipeline for given dataset
    if dataset_name == "sephora":
        X_train = df_train.drop(columns=[class_column])
        X_test = df_test.drop(columns=[class_column])
        pipeline = build_sephora_pipeline(model, vectorizer, feature_engineering)
    elif dataset_name == "movies":
        X_train = df_train["combined_text"]
        X_test = df_test["combined_text"]
        pipeline = build_movies_pipeline(model, vectorizer, feature_engineering)
    else:
        raise ValueError("Wrong dataset name")

    # Train the model and make predictions
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Compute evaluation metrics
    metrics = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Params": model_params,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="binary", zero_division=0),
    }

    # Display results
    print("Metryki testowe:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save results to JSON file
    out_file = f"results_final/{dataset_name}_{model_name}_final_metrics.json"
    os.makedirs("results_final", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
