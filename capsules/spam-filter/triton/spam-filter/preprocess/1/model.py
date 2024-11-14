import triton_python_backend_utils as pb_utils
import torch
from transformers import DistilBertTokenizer
import numpy as np
import time


class TritonPythonModel:
    def __init__(self, tokenizer_path="/mnt/models/preprocess/1/tokenizer"):
        self.logger = pb_utils.Logger
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.logger.log_info(f"Tokenizer: distilbert-base-uncased")

    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            ## 512 Due to the model token input limit
            self.tokenizer(
                request.inputs["text"],
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )

            response = pb_utils.InferenceResponse(
                output_tensors={
                    "input_ids": pb_utils.Tensor(self.tokenizer["input_ids"]),
                    "attention_mask": pb_utils.Tensor(self.tokenizer["attention_mask"]),
                }
            )
            responses.append(response)

        end_time_batch = time.perf_counter()
        self.logger.log_info(f"Batch time: {end_time_batch - start_time_batch}")
        return responses

    def finalize(self):
        self.tokenizer = None
