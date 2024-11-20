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

        # Load the id2label mapping from config.json
        with open("/mnt/models/postprocess/1/config.json", "r") as file:
            config = json.load(file)
        id2label = config["id2label"]

        for request in requests:
            # Extract the logits tensor
            logits_tensor = pb_utils.get_input_tensor_by_name(request, "logits")
            if logits_tensor is None:
                raise ValueError("Missing input tensor named 'logits'.")

            # Convert logits tensor to a Torch tensor
            logits = torch.tensor(logits_tensor.as_numpy())

            # Compute softmax probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Determine the predicted label
            predicted_label_index = torch.argmax(probs, dim=1).item()
            predicted_label = id2label[
                str(predicted_label_index)
            ]  # Use str key for JSON compatibility

            # Create output tensors
            label_tensor = pb_utils.Tensor(
                "label", np.array([predicted_label.encode("utf-8")], dtype=object)
            )
            probs_tensor = pb_utils.Tensor(
                "score", probs.cpu().numpy().astype(np.float32)
            )

            # Create inference response
            response = pb_utils.InferenceResponse(
                output_tensors=[label_tensor, probs_tensor]
            )
            responses.append(response)

        end_time_batch = time.perf_counter()
        self.logger.log_info(f"Batch time: {end_time_batch - start_time_batch}")
        return responses

    def finalize(self):
        self.logger.log_info("Finalizing the postprocess model.")
