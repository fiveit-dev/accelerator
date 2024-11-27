import os
import boto3
import json
from tqdm.notebook import tqdm
from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,  # type: ignore
)
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

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

        return rows, label2id, id2label


class Trainer:
    def text_classification(
        self,
        dataset: pd.DataFrame,
        label2id: dict,
        id2label: dict,
        base_model: str,
        model_name: str,
        MLFLOW_EXPERIMENT: str = "default",
        MLFLOW_RUN_NAME: str = "default",
        test_size: float = 0.2,
    ):
        input_column_name = "text"

        df_train, df_test = train_test_split(dataset, test_size=test_size)

        ## Convert to Huggingface dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        def preprocess_function(examples):
            return tokenizer(examples[input_column_name], truncation=True)

        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_test = test_dataset.map(preprocess_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            conf_matrix = confusion_matrix(labels, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.ylabel("True Labels")
            plt.xlabel("Predicted Labels")

            # Save the confusion matrix plot
            conf_matrix_filepath = "confusion_matrix.png"
            plt.savefig(conf_matrix_filepath)
            plt.close()
            return metric.compute(predictions=predictions, references=labels)

        os.environ["MFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT
        lr = 2e-5
        train_batch_size = 8
        eval_batch_size = 8
        epochs = 2
        decay = 0.01
        eval_strategy = "epoch"
        log_strategy = "epoch"
        training_args = TrainingArguments(
            hub_model_id=model_name,
            output_dir=MLFLOW_RUN_NAME,
            learning_rate=lr,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=decay,
            evaluation_strategy=eval_strategy,
            logging_strategy=log_strategy,
        )
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))  # type: ignore

        return Trainer(  # type: ignore
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            eval_dataset=tokenized_test,
        )
