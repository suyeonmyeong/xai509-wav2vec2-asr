# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

_DEFAULT_HF_HOME = "/home/suyeon/Desktop/workspace/speech/.cache/huggingface"
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Third-party imports
from transformers import pipeline

import sample_util

db_top_dir = os.environ.get(
    "DB_TOP_DIR",
    "/home/suyeon/Desktop/workspace/speech/database",
)
model_checkpoint = os.environ.get(
    "MODEL_CHECKPOINT",
    "/home/suyeon/Desktop/workspace/speech/outputs/wav2vec2-baseline/checkpoint-2000",
)
result_dir = os.environ.get("RESULT_DIR", os.getcwd())
result_prefix = os.environ.get("RESULT_PREFIX", "test")

test_clean_top_dir = os.path.join(db_top_dir, "test-clean")
test_other_top_dir = os.path.join(db_top_dir, "test-other")

os.makedirs(result_dir, exist_ok=True)

test_clean_dataset = sample_util.make_dataset(test_clean_top_dir, False)
test_other_dataset = sample_util.make_dataset(test_other_top_dir, False)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_checkpoint,
    device=0 if os.environ.get("ASR_DEVICE", "cpu").lower() == "cuda" else -1,
)

# Function to write REF/HYP pairs to a file
def write_results(dataset, transcriber, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            ref = data["labels"]
            hyp = transcriber(data["input_values"])["text"]
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")  # double newline for readability

# Write test_clean_dataset
write_results(
    test_clean_dataset,
    transcriber,
    os.path.join(result_dir, f"{result_prefix}_clean_result.txt"),
)

# Write test_other_dataset
write_results(
    test_other_dataset,
    transcriber,
    os.path.join(result_dir, f"{result_prefix}_other_result.txt"),
)
