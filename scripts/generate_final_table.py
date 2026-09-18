import json
import os
import pandas as pd
from typing import Dict, Any, List

RESULTS_DIR = "results_final"

def simplify_params(model_name: str, params: Dict[str, Any]) -> str:
    """
    Extracts and formats a simplified string of the most relevant hyperparameters
    for each model type (Random Forest or SVM).
    """
    if model_name == "rf":
        keys = ["n_estimators", "max_depth"]
    elif model_name == "svm":
        keys = ["C", "kernel"]
    else:
        return ""
    return ", ".join(f"{k}={params[k]}" for k in keys if k in params)


def main() -> None:
    rows = []

    # Iterate through all JSON result files in the results directory
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue

        # Load evaluation results
        with open(os.path.join(RESULTS_DIR, filename)) as f:
            data = json.load(f)

        # Extract dataset name, model type, metrics, and simplified parameters
        dataset = data["Dataset"]
        model = data["Model"]
        metrics = {k: data[k] for k in ["Accuracy", "Precision", "Recall", "F1 Score"]}
        params = simplify_params(model, data["Params"])

        # Format row for output table
        rows.append(
            {
                "Dataset": dataset,
                "Model": model.title()
                .replace("Rf", "Random Forest")
                .replace("Svm", "SVM"),
                "Params": params,
                **metrics,
            }
        )

    # Create DataFrame and export as markdown table
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Dataset", "Model"], ascending=[False, True])
    with open("final_table.md", "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
