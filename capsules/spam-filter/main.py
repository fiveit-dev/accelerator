import argparse
import boto3
import mlflow


def retrieve_from_s3(bucket):
    print(f"Retrieving model from S3 bucket: {bucket}")


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
    parser.add_argument("bucket", type=str, help="Name of the S3 bucket.")
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
