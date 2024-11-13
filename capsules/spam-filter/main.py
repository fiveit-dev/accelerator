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
ALQUIMIA_BUCKET = "alquimia-capsule"
DATASET_PATH = "datasets/text-classification/spam-filter"
MODEL_PATH = "models/spam-filter"

os.environ["MLFLOW_EXPERIMENT_NAME"] = "alquimia-capsules"
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


def retrieve_dataset():
    pages = paginator.paginate(Bucket=ALQUIMIA_BUCKET, Prefix=DATASET_PATH)
    annotations = []
    try:
        for page in tqdm(pages, leave=False, desc="Going through pages"):
            for obj in tqdm(page["Contents"], leave=False, desc="Loading annotations"):
                data = s3.get_object(Bucket=ALQUIMIA_BUCKET, Key=obj.get("Key"))
                content = json.loads((data["Body"].read()).decode("utf-8"))
                task = content.get("task").get("data").get("text")
                result = content.get("result")[0].get("value").get("choices")
                annotations.append(
                    {
                        "data": {"text": task},
                        "predictions": [{"result": {"value": {"choices": result}}}],
                    }
                )
    except Exception as e:
        print(f"An error occurred: {e}")

    with open("annotations.json", "w") as f:
        json.dump(annotations, f)


def retrieve_model():
    run_name = "Spam filter base training"
    with mlflow.start_run(run_name=run_name):
        pages = paginator.paginate(Bucket=ALQUIMIA_BUCKET, Prefix=MODEL_PATH)
        try:
            for page in tqdm(pages, leave=False, desc="Going through pages"):
                for obj in tqdm(page["Contents"], leave=False, desc="Loading model"):
                    data = s3.get_object(Bucket=ALQUIMIA_BUCKET, Key=obj.get("Key"))
                    file_name = obj.get("Key").split("/")[-1]
                    # Save the file temporarily in local
                    with open(file_name, "wb") as f:
                        f.write(data["Body"].read())
                    if file_name == "model.onnx":
                        model = onnx.load_model("./model.onnx")
                        mlflow.onnx.log_model(
                            model, registered_model_name="spam-filter"
                        )  ## Model registry
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
                        print(f"Uploading artifact: {file_name}")
                        mlflow.log_artifact(file_name)
                    os.remove(file_name)
            mlflow.end_run()
        except Exception as e:
            print(f"An error occurred: {e}")

    mlflow.end_run()


def retrieve_from_s3():
    retrieve_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve model from S3 and upload to MLflow."
    )

    args = parser.parse_args()

    retrieve_from_s3()
