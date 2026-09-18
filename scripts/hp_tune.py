import pandas as pd
import json
import yaml
import os
import argparse
from typing import Tuple, Dict, Any
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint, loguniform
from src.pipelines import build_movies_pipeline, build_sephora_pipeline
from sklearn.model_selection import train_test_split


def get_model_and_params_dist(name: str)-> Tuple[ClassifierMixin, Dict[str, Any]]:
    """
    Returns a model instance and the hyperparameter search space for RandomizedSearchCV.
    Supported models: RandomForest ("rf") and SVC ("svm").
    """
    if name == "rf":
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            "clf__n_estimators": randint(50, 200),
            "clf__max_depth": randint(5, 15),
            "clf__min_samples_split": randint(2, 10),
        }
    elif name == "svm":
        model = SVC(random_state=42)
        param_dist = {
            "clf__C": loguniform(1e-2, 1e2),
            "clf__gamma": loguniform(1e-4, 1e-1),
            "clf__kernel": ["linear", "rbf", "poly", "sigmoid"],
        }
    else:
        raise ValueError("Wrong model name")

    return model, param_dist


def main() -> None:
    # Load configuration from YAML
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    # Load parameters from configuration
    dataset_name = args.dataset
    model_name = args.model
    dataset_cfg = params["datasets"][dataset_name]
    class_column = dataset_cfg["label_column"]
    feature_engineering = params["train"]["feature_engineering"]
    vectorizer = params["train"]["vectorizer"]
    num_samples = params['tune']['tune_samples']
    n_iter = params['tune']['random_search_iter']

    df = pd.read_csv(f"/app/data/train_test/{dataset_name}_train_data_prepared.csv")
    df, _ = train_test_split(
                            df,
                            train_size=num_samples,
                            stratify=df[class_column],
                            random_state=42,
                        )

    # Skip tuning if model is dummy
    if model_name != "dummy":
        # Get model and hyperparameter distribution
        model, param_dist = get_model_and_params_dist(model_name)
        y = df[class_column]

        # Select features and build pipeline based on dataset
        if dataset_name == "sephora":
            X = df.drop(columns=[class_column])
            pipeline = build_sephora_pipeline(model, vectorizer, feature_engineering)

        elif dataset_name == "movies":
            X = df["combined_text"]
            pipeline = build_movies_pipeline(model, vectorizer, feature_engineering)
        else:
            raise ValueError("Wrong dataset name")

        # Perform randomized hyperparameter search
        search = RandomizedSearchCV(
            pipeline,
            param_dist,
            n_iter=n_iter,
            cv=5,
            scoring="f1",
            n_jobs=1,
            random_state=42,
            verbose=1,
        )
        res = search.fit(X, y)

        # Extract best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_

        # Save tuning results to JSON
        os.makedirs("results_hp_tuning", exist_ok=True)
        with open(
            f"results_hp_tuning/{dataset_name}_{model_name}_tuning.json", "w"
        ) as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "best_params": best_params,
                    "vectorizer": vectorizer,
                    "feature_engineering": feature_engineering,
                    "best_score": best_score,
                },
                f,
                indent=4,
            )
    else:
        # Save default dummy strategy if tuning is not needed
        os.makedirs("results_hp_tuning", exist_ok=True)
        with open(
            f"results_hp_tuning/{dataset_name}_{model_name}_tuning.json", "w"
        ) as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "best_params": {"clf__strategy": "most_frequent"},
                    "vectorizer": vectorizer,
                    "feature_engineering": feature_engineering,
                    "best_score": "N/A",
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    main()
