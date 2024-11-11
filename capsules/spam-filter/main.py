import argparse
import boto3
import mlflow
from botocore.config import Config
from botocore import UNSIGNED
from tqdm import tqdm

ALQUIMIA_S3 = "https://minio-api-minio.apps.alquimiaai.hostmydemo.online"
ALQUIMIA_BUCKET = "alquimia-capsules"
DATASET_PATH = "datasets/text-classification/spam-filter"


def retrieve_from_s3(bucket_to_store):
    s3 = boto3.client(
        "s3",
        endpoint_url=ALQUIMIA_S3,
        config=Config(signature_version=UNSIGNED),
    )

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=ALQUIMIA_BUCKET, Prefix=DATASET_PATH)
    try:
        for page in tqdm(pages, leave=False, desc="Going through pages"):
            for obj in tqdm(page["Contents"], leave=False, desc="Loading annotations"):
                data = s3.get_object(Bucket=ALQUIMIA_BUCKET, Key=obj.get("Key"))
                content = (data["Body"].read()).decode("utf-8")
    except Exception as e:
        print(f"An error occurred: {e}")


def upload_to_mlflow(serving_runtime, model_format, metrics):
    print(f"Uploading model with serving runtime: {serving_runtime}")
    print(f"Model format: {model_format}")
    if metrics:
        print("Logging metrics is enabled.")
    else:
        print("Logging metrics is disabled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve model from S3 and upload to MLflow."
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="Name of the S3 bucket where all datasets are going to be stored",
    )
    parser.add_argument(
        "--serving_runtime",
        type=str,
        required=True,
        help="Serving runtime for the model.",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        required=True,
        help="Model format (e.g., 'ONNX', 'TorchScript').",
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Enable logging of metrics."
    )

    args = parser.parse_args()

    retrieve_from_s3(args.bucket)

    upload_to_mlflow(args.serving_runtime, args.model_format, args.metrics)
