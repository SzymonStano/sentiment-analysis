import pandas as pd
import yaml
import argparse


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset_name = args.dataset

    df_train = pd.read_csv(
        f"/app/data/train_test/{dataset_name}_train_data.csv", engine="python"
    )
    df_test = pd.read_csv(
        f"/app/data/train_test/{dataset_name}_test_data.csv", engine="python"
    )

    # based on EDA
    if dataset_name == "sephora":
        cols_to_drop = params["clean"]["sephora"]["drop_columns"]

        df_train = df_train.drop(columns=cols_to_drop)
        df_test = df_test.drop(columns=cols_to_drop)

        # based on EDA on train dataset
        df_train["helpfulness"] = df_train["helpfulness"].fillna(0.5)
        df_test["helpfulness"] = df_test["helpfulness"].fillna(0.5)

        df_train["review_title"] = df_train["review_title"].fillna("")
        df_test["review_title"] = df_test["review_title"].fillna("")

        df_train["review_text"] = df_train["review_text"].fillna("")
        df_test["review_text"] = df_test["review_text"].fillna("")

        cols1 = ["skin_tone", "eye_color", "skin_type", "hair_color"]
        for col in cols1:
            df_train[col] = df_train[col].fillna("Unknown")
            df_test[col] = df_test[col].fillna("Unknown")

        cols2 = ["highlights", "tertiary_category"]
        for col in cols2:
            df_train[col] = df_train[col].fillna("None")
            df_test[col] = df_test[col].fillna("None")

        df_train = df_train.dropna().reset_index(drop=True)
        df_test = df_test.dropna().reset_index(drop=True)

        df_train["combined_text"] = (
            df_train["review_text"]
            + " "
            + df_train["review_title"]
            + " "
            + df_train["highlights"]
        )
        df_test["combined_text"] = (
            df_test["review_text"]
            + " "
            + df_test["review_title"]
            + " "
            + df_test["highlights"]
        )

        text_col = ["review_text", "review_title", "highlights"]
        df_train = df_train.drop(columns=text_col)
        df_test = df_test.drop(columns=text_col)

    elif dataset_name == "movies":
        df_train["text"] = df_train["text"].fillna("None")
        df_test["text"] = df_test["text"].fillna("None")

        df_train = df_train.rename(columns={"text": "combined_text"})
        df_test = df_test.rename(columns={"text": "combined_text"})
    else:
        raise ValueError("Wrong dataset name!")

    df_train.to_csv(
        f"/app/data/train_test/{dataset_name}_train_data_prepared.csv", index=False
    )
    df_test.to_csv(
        f"/app/data/train_test/{dataset_name}_test_data_prepared.csv", index=False
    )

if __name__ == "__main__":
    main()
