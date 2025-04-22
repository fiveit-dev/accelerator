import triton_python_backend_utils as pb_utils  # type: ignore[import]
import numpy as np
import time
import json
from helpers.engine import OrpheusModel
import threading
import os


class TritonPythonModel:
    def initialize(self, args):
        self.model_path = os.environ["MODEL_PATH"]
        self.logger = pb_utils.Logger
        self.model_config = model_config = json.loads(args["model_config"])
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )
        with open(f"{self.model_path}/1/config.json", "r") as f:
            self.config_dict = json.load(f)
        self.model = OrpheusModel(**self.config_dict["vllm"])
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def response_thread(self, request, text_input, speaker_input):
        try:
            syn_tokens = self.model.generate_speech(
                prompt=text_input, voice=speaker_input
            )
            channels = self.config_dict["model"]["channels"]
            sample_rate = self.config_dict["model"]["sample_rate"]
            sample_width = self.config_dict["model"]["sample_width"]

            for audio_chunk in syn_tokens:
                frame_count = len(audio_chunk) // (sample_width * channels)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "audio",
                            audio_array.reshape(frame_count, channels).astype(np.int16),
                        ),
                        pb_utils.Tensor(
                            "sample_rate",
                            np.array([sample_rate], dtype=np.int32),
                        ),
                    ]
                )
                request.send_response(response)

        except Exception as e:
            self.logger.log_error(f"Error: {e}")
        finally:
            # signal() semaphore to release the inflight thread
            with self.inflight_thread_count_lck:
                self.inflight_thread_count -= 1

    def execute(self, requests):
        for request in requests:
            text_input = (
                pb_utils.get_input_tensor_by_name(request, "text")
                .as_numpy()[0]
                .decode("utf-8")
            )
            speaker_input = (
                pb_utils.get_input_tensor_by_name(request, "speaker_id")
                .as_numpy()[0]
                .decode("utf-8")
            )
            self.logger.log_info(f"Input text: {text_input}")
            self.logger.log_info(f"Input speaker: {speaker_input}")
            thread = threading.Thread(
                target=self.response_thread,
                args=(request.get_response_sender(), text_input, speaker_input),
            )
            thread.daemon = True
            # wait() for the sempare to be used
            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1

            thread.start()

    def finalize(self):
        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = logging_time_sec / sleep_time_sec

        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
                if cycles % cycle_to_log == 0:
                    print(
                        f"Waiting for {self.inflight_thread_count} response threads to complete..."
                    )
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1
        self.logger.log_info("Finalize complete...")
