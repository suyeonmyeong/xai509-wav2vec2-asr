# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

_DEFAULT_HF_HOME = "/home/suyeon/Desktop/workspace/speech/.cache/huggingface"
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Third-party imports
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from transformers import AutoProcessor
import torch
import torch.nn.functional as F
import numpy as np
from jiwer import wer

# Custom imports
import sample_util


def _get_int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _get_float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def _get_report_to() -> List[str]:
    value = os.environ.get("REPORT_TO", "wandb")
    if value.lower() in {"", "none", "off", "false", "0"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


run_name = os.environ.get("RUN_NAME", "baseline_lr1e-4")
db_top_dir = os.environ.get(
    "DB_TOP_DIR",
    "/home/suyeon/Desktop/workspace/speech/database",
)
train_top_dir = os.path.join(db_top_dir, "libri_light/1h")
val_fraction = _get_float_env("VAL_FRACTION", 0.1)
train_shuffle_buffer = _get_int_env("TRAIN_SHUFFLE_BUFFER", 0)
train_split = os.environ.get("TRAIN_SPLIT", "train")
eval_split = os.environ.get("EVAL_SPLIT", "val")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

train_dataset = sample_util.make_dataset(
    train_top_dir,
    split=train_split,
    val_fraction=val_fraction,
    shuffle_buffer=train_shuffle_buffer,
)
val_dataset = None
if eval_split.lower() not in {"", "none", "off", "false", "0"}:
    val_dataset = sample_util.make_dataset(
        train_top_dir,
        split=eval_split,
        val_fraction=val_fraction,
    )


def compute_metrics(pred) -> Dict[str, float]:
    """Compute word error rate (WER) between predictions and labels.

    This function decodes the model's predicted token IDs and ground truth
    label IDs into strings, replacing ignored label tokens with the padding
    token ID. Then it computes WER using the `evaluate` library.

    Args:
        pred: A prediction object with attributes:
            - predictions: logits or probabilities of shape
                (batch_size, seq_len, vocab_size).
            - label_ids: ground truth token IDs with padding replaced by -100.

    Returns:
        Dict[str, float]: Dictionary with WER under the key 'wer'.
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 in labels with tokenizer pad token ID to enable decoding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_score = wer(label_str, pred_str)

    return {"wer": wer_score}


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that dynamically pads input values and labels for CTC training.

    This class pads the input audio features and the corresponding label sequences
    (token IDs) to the length of the longest element in the batch. It also replaces
    padding tokens in the labels with -100 to ensure they are ignored during the loss
    computation, as required by PyTorch's CTC loss implementation.

    Attributes:
        processor (AutoProcessor): The processor used for feature extraction and tokenization.
        padding (Union[bool, str]): Padding strategy. Defaults to "longest" to pad to the
            longest sequence in the batch.
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Pad inputs and labels in a batch for model training.

        Args:
            features: A list of feature dictionaries, each containing:
                - "input_values": the audio features (list or tensor).
                - "labels": the tokenized label sequence.

        Returns:
            A dictionary with padded input tensors and labels ready for the model:
            - "input_values": Padded input audio feature tensor.
            - "labels": Padded label tensor with padding tokens replaced by -100.
        """
        # Separate the input audio features and label sequences from the batch.
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Use the processor's pad method to pad input audio features to the same length.
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Pad the label sequences separately using the processor's pad method.
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding tokens in labels with -100 so that the loss function ignores them.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Add the processed labels to the batch dictionary.
        batch["labels"] = labels

        return batch


class CTCMaximumEntropyTrainer(Trainer):
    """Trainer with optional maximum entropy regularization."""

    def __init__(self, *args, max_entropy_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_entropy_weight = max_entropy_weight

    def _compute_mean_entropy(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        if attention_mask is None:
            return entropy.mean()

        input_lengths = attention_mask.sum(dim=-1)
        frame_lengths = self.model._get_feat_extract_output_lengths(input_lengths)
        time_steps = logits.size(1)
        frame_mask = (
            torch.arange(time_steps, device=logits.device)
            .unsqueeze(0)
            .expand(logits.size(0), -1)
        ) < frame_lengths.unsqueeze(1)

        entropy = entropy.masked_fill(~frame_mask, 0.0)
        normalizer = frame_mask.sum().clamp(min=1)
        return entropy.sum() / normalizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.max_entropy_weight > 0.0:
            entropy = self._compute_mean_entropy(
                outputs.logits,
                inputs.get("attention_mask"),
            )
            loss = loss - (self.max_entropy_weight * entropy)

        if return_outputs:
            return loss, outputs
        return loss



def _freeze_encoder_layers(model, num_layers: int) -> None:
    if num_layers <= 0:
        return
    encoder = getattr(getattr(model, "wav2vec2", None), "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError("Could not find wav2vec2.encoder.layers to freeze")
    for layer in layers[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def _configure_spec_augment(model) -> None:
    config = model.config
    config.apply_spec_augment = _get_bool_env(
        "APPLY_SPEC_AUGMENT",
        getattr(config, "apply_spec_augment", True),
    )
    config.mask_time_prob = _get_float_env(
        "MASK_TIME_PROB",
        getattr(config, "mask_time_prob", 0.05),
    )
    config.mask_time_length = _get_int_env(
        "MASK_TIME_LENGTH",
        getattr(config, "mask_time_length", 10),
    )
    config.mask_feature_prob = _get_float_env(
        "MASK_FEATURE_PROB",
        getattr(config, "mask_feature_prob", 0.0),
    )
    config.mask_feature_length = _get_int_env(
        "MASK_FEATURE_LENGTH",
        getattr(config, "mask_feature_length", 10),
    )
    config.layerdrop = _get_float_env("LAYERDROP", getattr(config, "layerdrop", 0.0))
    config.ctc_zero_infinity = _get_bool_env(
        "CTC_ZERO_INFINITY",
        getattr(config, "ctc_zero_infinity", False),
    )

# Instantiate the data collator for CTC loss with padding support.
# It dynamically pads the inputs and labels in each batch to the longest
# sequence, enabling efficient batch processing without manual padding.
data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding="longest"
)

# Load the pretrained Wav2Vec2 model with CTC (Connectionist Temporal Classification)
# head for speech recognition.
# - ctc_loss_reduction="mean" averages the CTC loss over the batch.
# - pad_token_id is set to the tokenizer's pad token to ensure correct masking.
model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)
_configure_spec_augment(model)

if _get_bool_env("FREEZE_FEATURE_ENCODER", False):
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()

_freeze_encoder_layers(model, _get_int_env("FREEZE_ENCODER_LAYERS", 0))

max_entropy_weight = _get_float_env("MAX_ENTROPY_WEIGHT", 0.0)

# Define the training arguments for the Hugging Face Trainer.
# These control training hyperparameters and runtime behavior:
training_args = TrainingArguments(
    # Directory to save model checkpoints and outputs.
    output_dir=os.environ.get(
        "OUTPUT_DIR",
        f"/home/suyeon/Desktop/workspace/speech/outputs/{run_name}",
    ),

    # Directory for local TensorBoard-compatible logs.
    logging_dir=os.environ.get(
        "LOGGING_DIR",
        f"/home/suyeon/Desktop/workspace/speech/logs/tensorboard/{run_name}",
    ),

    # Batch size per device (GPU/CPU) for training.
    per_device_train_batch_size=_get_int_env("TRAIN_BATCH_SIZE", 64),

    # Number of batches to accumulate gradients over before updating model weights.
    gradient_accumulation_steps=_get_int_env("GRAD_ACCUM_STEPS", 1),

    # Initial learning rate for the optimizer.
    learning_rate=_get_float_env("LEARNING_RATE", 1e-4),

    # Number of warmup steps to gradually increase learning rate at start.
    warmup_steps=_get_int_env("WARMUP_STEPS", 500),

    # Total number of training steps.
    max_steps=_get_int_env("MAX_STEPS", 2000),

    # Enable gradient checkpointing to reduce memory usage at the cost of extra compute.
    gradient_checkpointing=_get_bool_env("GRADIENT_CHECKPOINTING", True),

    # Use mixed precision training (float16) to speed up training and reduce memory.
    fp16=_get_bool_env("FP16", torch.cuda.is_available()),

    # Blackwell/Ampere GPUs can use bf16 for more stable mixed precision if desired.
    bf16=_get_bool_env("BF16", False),

    # Performs evaluation every N steps when an eval split is enabled.
    eval_strategy="steps" if val_dataset is not None else "no",

    # Batch size per device during evaluation.
    per_device_eval_batch_size=_get_int_env("EVAL_BATCH_SIZE", 24),

    # Save model checkpoints every N steps.
    save_steps=_get_int_env("SAVE_STEPS", 2000),

    # Keep only a few checkpoints so broad sweeps do not fill the disk.
    save_total_limit=_get_int_env("SAVE_TOTAL_LIMIT", 3),

    # Run evaluation every N steps during training.
    eval_steps=_get_int_env("EVAL_STEPS", 100),

    # Log training progress every N steps.
    logging_steps=_get_int_env("LOGGING_STEPS", 25),

    # Keep experiment tracking configurable: "wandb", "tensorboard", or "wandb,tensorboard".
    report_to=_get_report_to(),

    # Human-readable experiment name shown in W&B/TensorBoard.
    run_name=run_name,

    # Load the best model (lowest WER) at the end of training automatically.
    load_best_model_at_end=(
        _get_bool_env("LOAD_BEST_MODEL_AT_END", True)
        if val_dataset is not None
        else False
    ),

    # Metric to use for selecting the best model checkpoint.
    metric_for_best_model="wer",

    # Indicates that a lower metric score (WER) is better.
    greater_is_better=False,

    # Disable pushing model to the Hugging Face hub.
    push_to_hub=False,
)

# Create the Trainer instance to handle training and evaluation.
# This ties together the model, datasets, tokenizer, data collator, and metrics.
trainer = CTCMaximumEntropyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    max_entropy_weight=max_entropy_weight,
)

trainer.train()
