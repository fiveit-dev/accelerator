import triton_python_backend_utils as pb_utils
import torch
import numpy as np
import time
import json


class TritonPythonModel:
    def __init__(self):
        self.logger = pb_utils.Logger

    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            ## Get logits
            logits = request.inputs["logits"]
            with open("./config.json", "r") as file:
                config = json.load(file)

            id2label = config["id2label"]
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1)
            label = id2label[predicted_label.item()]
            response = pb_utils.InferenceResponse(
                output_tensors={
                    "label": pb_utils.Tensor(label),
                    "probs": pb_utils.Tensor(probs.cpu().numpy()),
                }
            )
            responses.append(response)

        end_time_batch = time.perf_counter()
        self.logger.log_info(f"Batch time: {end_time_batch - start_time_batch}")
        return responses

    def finalize(self):
        self.tokenizer = None
