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
            # Log the inputs for debugging
            self.logger.log_info(
                f"Received inputs: {[tensor.name() for tensor in request.inputs()]}"
            )

            # Extract the "input" tensor
            text_input_tensor = pb_utils.get_input_tensor_by_name(request, "input")

            if text_input_tensor is None:
                raise ValueError("Missing input tensor named 'input'.")

            # Convert the tensor to a string (assuming batch size 1)
            text_input = text_input_tensor.as_numpy()[0].decode("utf-8")

            # Tokenize the input text
            tokenized_output = self.tokenizer(
                text_input,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Prepare the output tensors as INT64
            input_ids_tensor = pb_utils.Tensor(
                "input_ids", tokenized_output["input_ids"].astype(np.int64)
            )
            attention_mask_tensor = pb_utils.Tensor(
                "attention_mask", tokenized_output["attention_mask"].astype(np.int64)
            )

            # Create and append the inference response
            response = pb_utils.InferenceResponse(
                output_tensors=[input_ids_tensor, attention_mask_tensor]
            )
            responses.append(response)

        end_time_batch = time.perf_counter()
        self.logger.log_info(f"Batch time: {end_time_batch - start_time_batch}")
        return responses

    def finalize(self):
        self.tokenizer = None
