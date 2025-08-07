import os
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from src.helpers.feature_engineering import compute_simple_features
from src.helpers.params.preprocessing_params import PreprocessingParams

preprocessing_params = PreprocessingParams()


def prepare_data():
    df = pd.read_csv(preprocessing_params.raw_s3_uri)[["URL", "label"]]
    features_df = df["URL"].apply(compute_simple_features).apply(pd.Series)
    df_processed = pd.concat([df[["label"]], features_df], axis=1).drop(columns="URL")

    X, y = df_processed.drop(columns=["label"]), df_processed["label"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=preprocessing_params.test_size,
        stratify=y,
        random_state=preprocessing_params.random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=preprocessing_params.val_size,
        stratify=y_temp,
        random_state=preprocessing_params.random_state,
    )

    os.makedirs(preprocessing_params.processed_local_dir, exist_ok=True)

    def save_split(X, y, name):
        df_out = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        df_out.dropna(how="all").to_csv(
            f"{preprocessing_params.processed_local_dir}/{name}.csv",
            index=False,
            header=False,
        )

    save_split(X_train, y_train, "train")
    save_split(X_val, y_val, "validation")
    save_split(X_test, y_test, "test")
    print("Saved splits locally")


def upload_to_s3():
    s3 = boto3.client("s3")
    for split in ["train", "validation", "test"]:
        local_path = f"{preprocessing_params.processed_local_dir}/{split}.csv"
        s3_key = f"{preprocessing_params.processed_s3_prefix}/{split}.csv"
        s3.upload_file(local_path, preprocessing_params.bucket, s3_key)
        print(f"Uploaded {split}.csv to s3://{preprocessing_params.bucket}/{s3_key}")


if __name__ == "__main__":
    prepare_data()
    upload_to_s3()
