import triton_python_backend_utils as pb_utils
import torch
from transformers import DistilBertTokenizer,
import numpy as np

class TritonPythonModel:
    def __init__(self,  tokenizer_path="/mnt/models/preprocess/1/tokenizer"):
        self.logger = pb_utils.Logger
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.logger.log_info(f"Speakers: {self.speakers}")
        
