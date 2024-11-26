import os
import boto3
import json
from tqdm.notebook import tqdm
from enum import Enum

S3_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT")

session = boto3.Session()
s3_client = session.client(
    "s3",
    region_name="nyc3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_KEY_ID,
    aws_secret_access_key=S3_SECRET_KEY,
)


class LabelStudio(str, Enum):
    text_classification = "text-classification"
    token_classification = "token-classification"
    ilab = "ilab"




class Retriever:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.paginator = s3_client.get_paginator("list_objects_v2")

    def label_studio(self, dataset_type: LabelStudio, dataset_name: str):
        ## TODO: Refactor this
        if dataset_type == LabelStudio.text_classification:
            return self.__text_classification_load_dataset(dataset_name)
        else:
            raise ValueError("Invalid dataset type")
            
    def __text_classification_load_dataset(self, dataset_name: str):
        """
        Load a text classification dataset from label studio
        """
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=f"datasets/labeled/text-classification/{dataset_name}",
        )
        rows = []
        id2label = {}
        label2id = {}

        i = 0
        j = 0

        for page in pages:
            for obj in tqdm(page["Contents"], leave=False, desc="Loading annotations"):
                data = s3_client.get_object(Bucket=self.bucket_name, Key=obj.get("Key"))
                content = (data["Body"].read()).decode("utf-8")
                if i > 0 and isinstance(content, str):
                    annotation = json.loads(content)
                    result = annotation.get("result")
                    if len(result) <= 0:
                        continue
                    label = ((result[0]).get("value")).get("choices")[0]

                    if label not in label2id:
                        label2id[label] = j
                        id2label[j] = label
                        j += 1
                    row = {
                        (annotation.get("result")[0]).get("from_name"): label,
                        **(annotation.get("task")).get("data"),
                    }
                    rows.append(row)
                i += 1

        return rows,label2id,id2label


class Trainer:
    pass
