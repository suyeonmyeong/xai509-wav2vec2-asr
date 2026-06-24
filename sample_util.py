# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import hashlib
import io
import os
from typing import Dict

_DEFAULT_HF_HOME = "/home/suyeon/Desktop/workspace/speech/.cache/huggingface"
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Third-party imports
import soundfile as sf
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


def _is_validation_sample(sample_key: str, val_fraction: float) -> bool:
    """Assign samples to validation deterministically from their key."""
    if val_fraction <= 0.0:
        return False
    digest = hashlib.md5(sample_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction

def preprocess_sample(sample: Dict, do_tokenization: bool = True) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'wav' (raw audio bytes)
            and 'txt' (transcript bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    waveform, sample_rate = sf.read(io.BytesIO(sample["audio"]), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    input_values = processor.feature_extractor(
        waveform, sampling_rate=sample_rate
    ).input_values[0]

    if isinstance(sample["text"], bytes):
        text = sample["text"].decode("utf-8").strip()
    else:
        text = sample["text"].strip()

    if do_tokenization:
        labels = processor.tokenizer(text).input_ids
    else:
        labels = text

    return {
        "input_values": input_values,
        "labels": labels,
        "__key__": sample["__key__"],
    }


def make_dataset(
    data_dir: str,
    do_tokenization: bool = True,
    split: str = "all",
    val_fraction: float = 0.1,
    shuffle_buffer: int = 0,
) -> wds.WebDataset:
    """Create a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    if split not in {"all", "train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    def _select_split(sample: Dict) -> bool:
        sample_key = sample["__key__"]
        is_val = _is_validation_sample(sample_key, val_fraction)
        if split == "all":
            return True
        if split == "train":
            return not is_val
        return is_val

    dataset = (
        wds.WebDataset(
            glob.glob(os.path.join(data_dir, "shard-*.tar")),
            shardshuffle=False,
        )
        .decode()
        .select(_select_split)
    )

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.map(lambda sample: preprocess_sample(sample, do_tokenization))
    return dataset
