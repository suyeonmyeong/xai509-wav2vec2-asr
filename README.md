# Wav2Vec2 ASR Fine-Tuning

This directory contains the Step 01 code for fine-tuning a pretrained Wav2Vec 2.0 model on a LibriSpeech WebDataset subset and evaluating word error rate (WER) on `test-clean` and `test-other`.

## Files

| File | Description |
|---|---|
| `sample_util.py` | WebDataset data loading and preprocessing utilities |
| `wav2vec_finetuning.py` | Wav2Vec2 CTC fine-tuning code |
| `wav2vec_inference.py` | ASR inference on `test-clean` and `test-other` |
| `evaluate_wer.py` | WER computation from REF/HYP result files |
| `results/` | Final WER files and REF/HYP outputs for the best model |

## Requirements

The code was run with Python 3.10. Main packages:

```bash
pip install torch torchaudio transformers webdataset soundfile jiwer numpy wandb
```

`wandb` is optional. To disable experiment logging, set `REPORT_TO=none`.

## Data Layout

Set `DB_TOP_DIR` to the directory containing the WebDataset shards:

```text
DB_TOP_DIR/
  libri_light/1h/
    shard-*.tar
  test-clean/
    shard-*.tar
  test-other/
    shard-*.tar
```

Do not extract the individual shard `.tar` files. They are read directly by WebDataset.

## Training

Baseline-style run:

```bash
DB_TOP_DIR=/path/to/database \
RUN_NAME=baseline_lr1e-4 \
OUTPUT_DIR=./outputs/baseline_lr1e-4 \
REPORT_TO=none \
python wav2vec_finetuning.py
```

Final regularized setting used in my experiments:

```bash
DB_TOP_DIR=/path/to/database \
RUN_NAME=final_freeze3_shuffle_specaug_maxent \
OUTPUT_DIR=./outputs/final_freeze3_shuffle_specaug_maxent \
REPORT_TO=none \
TRAIN_SPLIT=all \
EVAL_SPLIT=none \
TRAIN_BATCH_SIZE=32 \
EVAL_BATCH_SIZE=32 \
LEARNING_RATE=1e-4 \
MAX_STEPS=2000 \
WARMUP_STEPS=500 \
FREEZE_ENCODER_LAYERS=3 \
TRAIN_SHUFFLE_BUFFER=512 \
MASK_TIME_PROB=0.075 \
MASK_FEATURE_PROB=0.03 \
MAX_ENTROPY_WEIGHT=0.001 \
BF16=true \
FP16=false \
python wav2vec_finetuning.py
```

The default loss is CTC. When `MAX_ENTROPY_WEIGHT` is greater than `0`, the training objective becomes CTC loss plus maximum entropy regularization.

## Inference and WER

Run inference:

```bash
DB_TOP_DIR=/path/to/database \
MODEL_CHECKPOINT=./outputs/final_freeze3_shuffle_specaug_maxent/checkpoint-2000 \
RESULT_DIR=./results \
RESULT_PREFIX=final \
ASR_DEVICE=cuda \
python wav2vec_inference.py
```

Compute WER:

```bash
python evaluate_wer.py ./results/final_clean_result.txt
python evaluate_wer.py ./results/final_other_result.txt
```

## Main Result

| Model | test-clean WER | test-other WER |
|---|---:|---:|
| Baseline | 22.86% | 32.48% |
| Final regularized model | 20.75% | 28.62% |

Lower WER is better.

The final REF/HYP outputs and WER text files are included in `results/`.
