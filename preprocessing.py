from datetime import datetime
import json
import re
import os
import logging
import subprocess
import sys
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import (
    Wav2Vec2Processor,
    WhisperProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from datasets import DatasetDict, Dataset, load_dataset
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging_level = logging.DEBUG
logger.setLevel(logging_level)


class AudioConfig:
    sr = 16000
    duration = 30
    max_audio_len = sr * duration


def clean_text(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    # remove multiple spaces
    text = re.sub(r"\s\s+", " ", text)
    # strip trailing spaces
    text = text.strip()
    text = text.replace(">", "")
    text = text.replace("\t", " ")
    text = text.replace("\n", "")
    text = text.lower()
    text = (
        text.replace(" comma,", ",")
        .replace(" koma,", " ")
        .replace(" coma,", " ")
        .replace(" full stop.", ".")
        .replace(" full stop", ".")
        .replace(",.", ".")
        .replace(",,", ",")
        .strip()
    )
    text = " ".join(text.split())
    # text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", "", text)
    return text


def pad_zeros(x, size, sr):
    return np.pad(x, (0, max(0, size - len(x))), "constant") if len(x) < size else x


def load_vocab(model_path, checkpoints_path, exp_dir, raw_datasets):
    create_new_vocab = False
    vocab_file_name = None

    if os.path.isdir(model_path) and "vocab.json" in os.listdir(model_path):
        vocab_files = [
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "special_tokens_map.json",
        ]
        for v in vocab_files:
            subprocess.call(
                ["cp", os.path.join(model_path, v), os.path.join(checkpoints_path, v)]
            )
        vocab_file_name = os.path.join(checkpoints_path, "vocab.json")
        if os.path.isfile(vocab_file_name):
            print(f"vocab detected at {vocab_file_name}")
        else:
            create_new_vocab = True

    elif os.path.isdir(checkpoints_path) and len(os.listdir(checkpoints_path)) > 0:
        vocab_file_name = [x for x in os.listdir(checkpoints_path) if "vocab" in x]
        if len(vocab_file_name) > 0:
            vocab_file_name = os.path.join(checkpoints_path, vocab_file_name[0])
            print(f"vocab detected at {vocab_file_name}")
        else:
            create_new_vocab = True
    else:
        create_new_vocab = True

    if create_new_vocab:
        vocab_dict = create_vocab(raw_datasets)
        vocab_file_name = f'vocab-{datetime.now().strftime("%d-%m-%Y--%H:%M:%S")}.json'
        vocab_file_name = os.path.join(exp_dir, "checkpoints", vocab_file_name)
        logger.debug(f"creating new vocab {vocab_file_name}")
        with open(vocab_file_name, "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
    elif vocab_file_name:
        with open(vocab_file_name, "r") as vocab_file:
            vocab_dict = json.load(vocab_file)
    else:
        vocab_dict = {}

    logger.info(f"---vocab dict: {len(vocab_dict)}\n{vocab_dict}")
    return vocab_file_name


def load_audio_file(file_path):
    try:
        data, sr = librosa.core.load(file_path, sr=AudioConfig.sr)
        if sr != AudioConfig.sr:
            data = librosa.resample(data, sr, AudioConfig.sr)
        if len(data) < sr:
            data = pad_zeros(data, AudioConfig.sr, AudioConfig.sr)
    except Exception as e:
        print(f"{file_path} not found {str(e)}")
        data = np.random.rand(AudioConfig.sr * 3).astype(np.float32)
    return data


def load_data(train_path, val_path, language=None):
    if language:
        data = load_dataset("csv", data_files={"train": train_path, "val": val_path})
        data_filtered = data.filter(lambda x: x["language"] == language)
        return data_filtered
    else:
        return load_dataset("csv", data_files={"train": train_path, "val": val_path})


def remove_special_characters(batch):
    batch["text"] = clean_text(batch["text"]) + " "
    return batch


def extract_chars_vocab(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def special_tokens(vocab_dict):
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


def create_vocab(raw_datasets):
    raw_datasets = raw_datasets.map(remove_special_characters, num_proc=6)
    vocabs = raw_datasets.map(
        extract_chars_vocab,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=raw_datasets.column_names["train"],
    )

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["val"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict = special_tokens(vocab_dict)
    return vocab_dict
