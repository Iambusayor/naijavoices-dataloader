import os, sys
import pandas as pd
from typing import Callable
from datasets import load_dataset, concatenate_datasets
import torch
from preprocessing import clean_text, load_audio_file, AudioConfig

import logging
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ignore specific warnings
warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging_level = logging.DEBUG
logger.setLevel(logging_level)


def load_data(train_path, val_path):
    return load_dataset("csv", data_files={"train": train_path, "val": val_path})


SKIP_AUDIO_FILES = [
    "20240422021820-84-1000-261484-shin-kun-ta-a-zuwa-london.wav",
    "20240319060704-196-3496-1280375-gaa-ulo-ahia-echi.wav",
    "20240109163555-58-1028-235602-w-n-f-r-n-orin-y-n-n-n--j-wa.wav",
    "20240129224707-98-1063-326868-ose-ose-na-ewe-ogologo-awa-iji.wav",
    "20240327201336-250-4824-957121-zaki-yi-rawa-in-an-kunna-wa-a.wav",
]


class NaijaVoices(torch.utils.data.Dataset):
    def __init__(
        self,
        data_file: str,
        max_audio_len_secs: int,
        audio_dir: str,
        processor: Callable,
        feature_extractor: Callable,
        tokenizer: Callable,
        language_iso: str = None,
        audio_col: str = "audio_path",
        label_col: str = "text",
    ) -> None:
        super().__init__()
        self.data_file = data_file
        self.audio_dir = audio_dir
        self.max_audio_len_secs = max_audio_len_secs
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_col = audio_col
        self.label_col = label_col
        data = pd.read_csv(self.data_file)
        if language_iso:
            data = data[data["language"].isin([language_iso])]

        logger.info(f"Dataset size: {data.shape}")
        data = data[data["duration"] <= self.max_audio_len_secs]
        data = data[~data[self.audio_col].isin(SKIP_AUDIO_FILES)]  # skip bad audios

        self.data = data.drop_duplicates(subset=["audio"])
        logger.info(f"Dataset size after deduplication: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file_path = os.path.join(
            self.audio_dir, self.data.iloc[idx][self.audio_col]
        )
        transcription = self.data.iloc[idx][self.label_col]
        result = self.transform_batch(audio_file_path, transcription)
        if isinstance(result, dict):
            return result

        return {
            "input_features": result[0],
            "input_lengths": len(result[0]),
            "labels": result[1],
        }

    def transform_batch(self, audio_path, text):
        # Load and resample audio data to 16KHz
        try:
            speech = load_audio_file(audio_path)
        except Exception as e:
            print(f"{audio_path} not found {str(e)}")

        # Compute log-Mel input features from input audio array
        audio = self.feature_extractor(speech, sampling_rate=AudioConfig.sr)
        # Check if input_features exists, otherwise use input_values
        if hasattr(audio, "input_features"):
            audio = audio.input_features[0]

            # Encode target text to label ids
            text = clean_text(text)
            # self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
            labels = self.tokenizer(text.lower()).input_ids
            return audio, labels
        else:
            input_values = audio.input_values[0]

            text = clean_text(text)
            labels = self.tokenizer(text.lower()).input_ids

            # Return the expected dictionary structure
            return {
                "input_values": input_values,
                "input_lengths": len(input_values),
                "labels": labels,
            }


class FleursDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, language_iso, max_audio_len_secs=30):
        # Load the Google Fleurs dataset for the specified language and split
        if language_iso != "all":
            data = load_dataset(
                "google/fleurs",
                f"{language_iso}_ng",
                split=f"{split}",
                trust_remote_code=True,
            )
        else:
            # get datasets for each language, and concatenate them
            data_ha = load_dataset(
                "google/fleurs",
                f"ha_ng",
                split=f"{split}",
                trust_remote_code=True,
            )

            data_ig = load_dataset(
                "google/fleurs",
                f"ig_ng",
                split=f"{split}",
                trust_remote_code=True,
            )

            data_yo = load_dataset(
                "google/fleurs",
                f"yo_ng",
                split=f"{split}",
                trust_remote_code=True,
            )

            data = concatenate_datasets([data_ha, data_ig, data_yo])
        self.processor = processor
        self.max_audio_len_secs = max_audio_len_secs

        # Filter based on audio duration (optional)
        self.language_iso = language_iso
        self.data = data.filter(
            lambda example: example["audio"]["sampling_rate"] == 16000
        )
        logger.info(f"Fluers Dataset size: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = self.transform_batch(
            audio_array=self.data["audio"][idx]["array"],
            text=self.data["transcription"][idx],
        )
        if isinstance(result, dict):
            return result

        return {
            "input_features": result[0],
            "input_lengths": len(result[0]),
            "labels": result[1],
        }

    def transform_batch(self, audio_array, text):

        # Compute log-Mel input features from input audio array
        audio = self.processor.feature_extractor(
            audio_array, sampling_rate=AudioConfig.sr
        )
        # Check if input_features exists, otherwise use input_values
        if hasattr(audio, "input_features"):
            audio = audio.input_features[0]

            # Encode target text to label ids
            text = clean_text(text)
            # self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
            labels = self.processor.tokenizer(text.lower()).input_ids
            return audio, labels
        else:
            input_values = audio.input_values[0]

            text = clean_text(text)
            labels = self.processor.tokenizer(text.lower()).input_ids

            # Return the expected dictionary structure
            return {
                "input_values": input_values,
                "input_lengths": len(input_values),
                "labels": labels,
            }
