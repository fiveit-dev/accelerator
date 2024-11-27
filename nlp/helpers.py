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
    Trainer,
)
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import random

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


class AlquimiaTrainer:

    def __compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
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
        lr: float = 2e-5,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        epochs: int = 2,
        decay: float = 0.01,
        eval_strategy: str = "epoch",
        log_strategy: str = "epoch",
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

        os.environ["MFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT
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

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
            compute_metrics=self.__compute_metrics,
            eval_dataset=tokenized_test,
        )


class Helper:
    def bprint(self, text):
        bold_text = f"\033[1m{text}\033[0m"
        print(bold_text)

    def __read_md(self, file_path):
        """Reads the content of a Markdown file."""
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return content

    def __read_docx(self, file_path):
        """Reads the content of a docx file."""
        from docx import Document

        doc = Document(file_path)
        text = ""

        for para in doc.paragraphs:
            text += para.text

        return text

    def __read_pdf(self, filepath):
        """Reads text from a PDF file."""
        from PyPDF2 import PdfReader

        text = ""
        reader = PdfReader(filepath)

        for page in reader.pages:
            text += page.extract_text()

        return text

    def read_file(self, file_path):
        """Calls the appropiate read function for the input-file extension."""
        extension = file_path.split(".")[-1]

        if extension == "md":
            content = self.__read_md(file_path)
        elif extension == "pdf":
            content = self.__read_pdf(file_path)
        elif extension == "docx":
            content = self.__read_docx(file_path)
        else:
            raise TypeError(
                f"File should be either md, pdf or docx. {extension} is not compatible."
            )

        return content

    def chunk_text(
        self, full_text, max_words=100, overlap=0, separator=r"(?<=[.!?])\s+"
    ):
        """
        Given a large string, returns a list of chunks based on a maximum word count, overlap, and separator pattern.

        Args:
        - full_text (str): The complete text to be chunked.
        - max_words (int): Maximum number of words per chunk.
        - overlap (int): Number of words to overlap between chunks.
        - separator (str): Regex pattern to split sentences based on punctuation.

        Returns:
        - List of text chunks.
        """
        import re

        # Split based on the separator, typically at sentence boundaries
        sentences = re.split(separator, full_text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            # Tokenize the sentence into words
            words = sentence.split()
            # Add to current chunk if within word limit
            if len(current_chunk) + len(words) <= max_words:
                current_chunk.extend(words)
            else:
                # Append the current chunk to chunks
                chunks.append(" ".join(current_chunk))
                # Start new chunk with an overlap only if overlap > 0
                current_chunk = (
                    current_chunk[-overlap:] if overlap > 0 else []
                ) + words

        # Append any remaining text as a final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
        # Find a random example in the selected topic's data

    def __find_random_value(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively search in nested dictionaries
                result = self.__find_random_value(value)
                if result:
                    return key, result
            elif isinstance(value, list):
                # Return a random item from the list
                return key, random.choice(value)
        return None

    def get_items_from_dict(self, data):
        # Choose a random top-level topic
        random_topic_key = random.choice(list(data.keys()))
        random_topic_data = data[random_topic_key]

        # Get the example key and value
        example_key, example_value = self.__find_random_value(random_topic_data)

        # Return using the dynamically found keys
        return {random_topic_key: random_topic_key, example_key: example_value}

    def get_random_items_from_json(self, data):
        """
        input:
        {
        "key1": [str, str, str, ...],
        "key2": [list, list, list, ...],
        "key3": [dict, dict, dict, ...],
        ...
        }
        output:
        {
        "key1": random string,
        "key2": random list,
        random dict,
        ...
        }
        """
        import random

        output = {}
        for key, list_ in data.items():
            if isinstance(list_, list):
                value = random.choice(list_)
                if isinstance(value, dict):
                    output.update(value)
                else:
                    output[key] = value
            else:
                raise TypeError("Expected dictionary with lists as values.")
        return output
