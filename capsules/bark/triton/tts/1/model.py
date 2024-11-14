import triton_python_backend_utils as pb_utils
import torch
from transformers import BarkModel, AutoProcessor
import numpy as np
import time
import io
import wave
from bark.generation import (
    load_model,
    codec_decode,
    _load_history_prompt,
    _tokenize,
    SAMPLE_RATE,
    SUPPORTED_LANGS,
)


class TritonPythonModel:
    def __init__(self, device="cuda", model_name="/mnt/models/bark/1/tts_model"):
        self.logger = pb_utils.Logger

        self.model = BarkModel.from_pretrained(model_name)
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load available speakers
        self.speakers = self.get_speakers()
        self.logger.log_info(f"Speakers: {self.speakers}")
        self.device = device

    def get_speakers(self):
        speakers = ["Unconditional", "Announcer"] + [
            f"Speaker {n} ({lang})" for lang, _ in SUPPORTED_LANGS for n in range(10)
        ]
        return {"speakers": speakers}

    def gen_tts(self, text, history_prompt):
        # Mapping speaker prompts to available options
        PROMPT_LOOKUP = {f"Speaker {n} (en)": f"en_speaker_{n}" for n in range(10)}
        PROMPT_LOOKUP["Unconditional"] = None
        PROMPT_LOOKUP["Announcer"] = "announcer"

        # Convert history prompt to the appropriate speaker preset
        history_prompt = PROMPT_LOOKUP.get(history_prompt, None)

        # Prepare the inputs and generate audio
        inputs = self.processor(text, voice_preset=history_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        audio_arr = self.model.generate(**inputs)
        audio_arr = audio_arr[0].cpu().numpy()
        audio_arr = np.int16(audio_arr * 32767)  # Convert to 16-bit PCM

        return 24000, audio_arr  # Bark uses 24kHz audio

    def audio_array_to_wav(self, audio_array, sample_rate=24000):
        """Convert a numpy audio array to WAV binary data."""
        # Create an in-memory BytesIO buffer
        audio_bytes = io.BytesIO()

        # Use the wave module to write the WAV format
        with wave.open(audio_bytes, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
            wf.setframerate(sample_rate)  # Set the sample rate
            wf.writeframes(audio_array.tobytes())  # Write the audio frames as bytes

        # Reset the buffer position to the beginning
        audio_bytes.seek(0)

        return audio_bytes

    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()

        for request in requests:
            start_time = time.perf_counter()

            # Retrieve text input and speaker selection
            text_input = (
                pb_utils.get_input_tensor_by_name(request, "text")
                .as_numpy()[0]
                .decode("utf-8")
            )
            speaker_input = (
                pb_utils.get_input_tensor_by_name(request, "speaker")
                .as_numpy()[0]
                .decode("utf-8")
            )

            # Generate TTS audio
            sampling_rate, audio_arr = self.gen_tts(text_input, speaker_input)

            # Convert audio array to binary WAV format
            audio_wav_bytes = self.audio_array_to_wav(
                audio_arr, sample_rate=sampling_rate
            )

            # Log the output text for debugging
            self.logger.log_info(
                f"Generated audio for text: '{text_input}' with speaker: '{speaker_input}'"
            )

            # Prepare inference response with audio binary data
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "audio", np.array(audio_wav_bytes.getvalue(), dtype=np.object_)
                    ),
                    pb_utils.Tensor(
                        "sampling_rate", np.array([sampling_rate], dtype=np.int32)
                    ),
                ]
            )

            self.logger.log_info(
                f"Time taken for single request: {time.perf_counter() - start_time}"
            )
            responses.append(inference_response)

        self.logger.log_info(
            f"Time taken by batch: {time.perf_counter() - start_time_batch}"
        )
        return responses

    def finalize(self, args):
        self.processor = None
        self.model = None
