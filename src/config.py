"""Configuration and project paths for Amharic GPT."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
VOCAB_DIR = MODELS_DIR / "vocab"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

DEFAULT_CLEANED_TEXT = PROCESSED_DATA_DIR / "cleaned_amh_data.txt"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "amharic_gpt.pt"
DEFAULT_LOSS_PLOT = LOGS_DIR / "loss_curve.png"

# --- Tokenizer ---
TOKENIZER_NAME = "xlm-roberta-base"

# --- Model architecture (~124M params, GPT-2 small style) ---
GPT_CONFIG_124M = {
    "vocab_size": 250002,    # XLM-R vocab; overwritten at runtime from the loaded tokenizer
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# --- Training defaults ---
TRAIN_DEFAULTS = {
    "n_epochs": 3,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    "eval_freq": 100,
    "eval_iter": 20,
    "warmup_steps": 20,
    "train_ratio": 0.9,
    "stride": 256,
}
