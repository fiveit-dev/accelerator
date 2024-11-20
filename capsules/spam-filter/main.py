import argparse
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from tqdm import tqdm
import json
import os
import mlflow
import onnx
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
import mlflow.data

## Retrieval data from alquimia capsule public S3
ALQUIMIA_S3 = "https://minio-api-minio.apps.alquimiaai.hostmydemo.online"
ALQUIMIA_BUCKET = "alquimia-capsules"
DATASET_PATH = "datasets/text-classification/spam-filter"
MODEL_PATH = "models/spam-filter"

os.environ["MLFLOW_EXPERIMENT_NAME"] = "alquimiai-capsules"
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")

# MLFLOW_TRACKING_USERNAME & MLFLOW_TRACKING_PASSWORD should be set by default
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


s3 = boto3.client(
    "s3",
    endpoint_url=ALQUIMIA_S3,
    config=Config(signature_version=UNSIGNED),
)
paginator = s3.get_paginator("list_objects_v2")


def retrieve_model():
    run_name = "Spam filter base training"
    with mlflow.start_run(run_name=run_name):
        # Retrieve and upload all relevant files and directories
        pages = paginator.paginate(Bucket=ALQUIMIA_BUCKET, Prefix=MODEL_PATH)
        try:
            # Log the entire folder structure under triton/spam-filter
            for subdir in [
                "ensemble_spam_filter",
                "postprocess",
                "preprocess",
                "model",
            ]:
                mlflow.log_artifacts(
                    os.path.join("triton/spam-filter", subdir),
                    artifact_path=subdir,  # Avoid repeating the parent directory name
                )

            for page in tqdm(pages, leave=False, desc="Going through pages"):
                for obj in tqdm(page["Contents"], leave=False, desc="Loading model"):
                    file_name = obj.get("Key").split("/")[-1]
                    data = s3.get_object(Bucket=ALQUIMIA_BUCKET, Key=obj.get("Key"))

                    # Save and upload config.json into the postprocess directory
                    if file_name == "config.json":
                        config_path = os.path.join(
                            "triton/spam-filter/postprocess", file_name
                        )
                        with open(config_path, "wb") as f:
                            f.write(data["Body"].read())
                        mlflow.log_artifact(config_path, artifact_path="postprocess/1")
                        os.remove(config_path)
                    else:
                        with open(file_name, "wb") as f:
                            f.write(data["Body"].read())

                        if file_name == "model.onnx":
                            model = onnx.load_model("./model.onnx")
                            mlflow.onnx.log_model(
                                model,
                                artifact_path="model/1",
                                registered_model_name="spam-filter",
                            )
                        elif file_name == "train_dataset.csv":
                            train_dataset_csv = pd.read_csv(file_name)
                            train_dataset: PandasDataset = mlflow.data.from_pandas(  # type: ignore
                                train_dataset_csv, source="Label Studio"
                            )
                            mlflow.log_input(train_dataset, context="training")
                        elif file_name == "test_dataset.csv":
                            test_dataset_csv = pd.read_csv(file_name)
                            test_dataset: PandasDataset = mlflow.data.from_pandas(  # type: ignore
                                test_dataset_csv, source="Label Studio"
                            )
                            mlflow.log_input(test_dataset, context="testing")
                        else:
                            mlflow.log_artifact(file_name)

                        os.remove(file_name)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            mlflow.end_run()


def retrieve_from_s3():
    retrieve_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve model from S3 and upload to MLflow."
    )

    args = parser.parse_args()

    retrieve_from_s3()
