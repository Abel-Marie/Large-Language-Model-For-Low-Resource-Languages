"""Amharic GPT — CLI entry point.

Examples:
    python main.py preprocess --input data/raw/amh.txt --output data/processed/cleaned.txt
    python main.py train --epochs 3 --batch-size 8 --lr 1e-4
    python main.py generate "ሰላም! ይህ የ" --max-new-tokens 100 --temperature 0.8 --top-k 35
"""
import argparse
import sys
from pathlib import Path

import torch

from src import config


def cmd_preprocess(args):
    from src.data_preprocessing import load_dataset, clean_corpus, save_cleaned

    df = load_dataset(args.input)
    cleaned = clean_corpus(
        df["text"].tolist(), min_length=args.min_len, verbose=args.verbose
    )
    save_cleaned(cleaned, args.output)
    print(f"Cleaned file saved to: {args.output}")


def cmd_train(args):
    import matplotlib.pyplot as plt
    from transformers import XLMRobertaTokenizerFast

    from src.dataset import create_dataloader
    from src.gpt_model import GPTModel
    from src.training import train_model

    cfg = dict(config.GPT_CONFIG_124M)
    train_cfg = dict(config.TRAIN_DEFAULTS)
    if args.epochs is not None:
        train_cfg["n_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_path = Path(args.data) if args.data else config.DEFAULT_CLEANED_TEXT
    if not text_path.exists():
        sys.exit(
            f"Cleaned text not found at {text_path}.\n"
            f"Run `python main.py preprocess --input <raw> --output {text_path}` first."
        )
    text_data = text_path.read_text(encoding="utf-8")
    print(f"Loaded {len(text_data):,} characters from {text_path}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(config.TOKENIZER_NAME)
    cfg["vocab_size"] = tokenizer.vocab_size
    print(f"Tokenizer: {config.TOKENIZER_NAME} | vocab size: {cfg['vocab_size']:,}")

    split = int(train_cfg["train_ratio"] * len(text_data))
    train_loader = create_dataloader(
        text_data[:split],
        tokenizer,
        batch_size=train_cfg["batch_size"],
        max_length=cfg["context_length"],
        stride=train_cfg["stride"],
        shuffle=True,
    )
    val_loader = create_dataloader(
        text_data[split:],
        tokenizer,
        batch_size=train_cfg["batch_size"],
        max_length=cfg["context_length"],
        stride=train_cfg["stride"],
        shuffle=False,
    )
    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")

    torch.manual_seed(123)
    model = GPTModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    print("Starting training...")
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        n_epochs=train_cfg["n_epochs"],
        eval_freq=train_cfg["eval_freq"],
        eval_iter=train_cfg["eval_iter"],
        warmup_steps=train_cfg["warmup_steps"],
        start_context="ሰላም! ይህ የ",
        tokenizer=tokenizer,
    )

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_ckpt) if args.out_ckpt else config.DEFAULT_CHECKPOINT
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")

    if train_losses:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(tokens_seen, train_losses, label="Training Loss")
        ax1.plot(tokens_seen, val_losses, linestyle="--", label="Validation Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Losses")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(tokens_seen, lrs, label="Learning Rate", color="green")
        ax2.set_xlabel("Tokens Seen")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("LR Schedule")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(config.DEFAULT_LOSS_PLOT)
        print(f"Saved loss curve to: {config.DEFAULT_LOSS_PLOT}")


def cmd_generate(args):
    from transformers import XLMRobertaTokenizerFast

    from src.gpt_model import GPTModel
    from src.training import generate, text_to_token_ids, token_ids_to_text

    ckpt_path = Path(args.ckpt) if args.ckpt else config.DEFAULT_CHECKPOINT
    if not ckpt_path.exists():
        sys.exit(
            f"Checkpoint not found at {ckpt_path}.\n"
            f"Run `python main.py train` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["cfg"]

    model = GPTModel(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(config.TOKENIZER_NAME)
    encoded = text_to_token_ids(args.prompt, tokenizer).to(device)
    out = generate(
        model,
        encoded,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(token_ids_to_text(out, tokenizer))


def build_parser():
    p = argparse.ArgumentParser(description="Amharic GPT — train and generate.")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Clean raw Amharic text.")
    pp.add_argument("--input", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--min-len", type=int, default=10)
    pp.add_argument("--verbose", action="store_true")
    pp.set_defaults(func=cmd_preprocess)

    pt = sub.add_parser("train", help="Train the Amharic GPT model.")
    pt.add_argument("--data", default=None, help="Path to cleaned text file.")
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--out-ckpt", default=None, help="Path to save checkpoint.")
    pt.set_defaults(func=cmd_train)

    pg = sub.add_parser("generate", help="Generate text from a checkpoint.")
    pg.add_argument("prompt", help="Amharic prompt to continue.")
    pg.add_argument("--ckpt", default=None)
    pg.add_argument("--max-new-tokens", type=int, default=100)
    pg.add_argument("--temperature", type=float, default=0.8)
    pg.add_argument("--top-k", type=int, default=35)
    pg.set_defaults(func=cmd_generate)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
