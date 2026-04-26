# Amharic GPT — A Low-Resource LLM From Scratch

> A GPT-style decoder-only Transformer trained **from scratch** in PyTorch for Amharic text generation.
> ~124M parameters, XLM-RoBERTa tokenizer, end-to-end pipeline from raw text cleaning through training and sampling.

Most foundation LLMs treat Amharic as a low-priority long-tail language. This repo:
- reimplements every component of a GPT — attention, feed-forward, layer norm, transformer block, generator — without using `transformers.AutoModel`,
- provides an Amharic-aware text-cleaning pipeline (Fidel character normalization, Geez numeral conversion, URL/handle stripping),
- runs end-to-end with a single CLI: `preprocess` → `train` → `generate`.

## What this does

- **Clean** raw Amharic text — Fidel character normalization, numeral conversion, URL/handle stripping, whitespace normalization. See `src/utils_text.py` and `src/data_preprocessing.py`.
- **Tokenize** with the multilingual **XLM-RoBERTa** tokenizer (broad coverage of Geez script).
- **Train** a **GPT-2-small architecture** decoder-only Transformer (12 layers, 12 heads, 768 emb dim, 256 context window) on the cleaned corpus, using **linear warmup + cosine LR decay**, **AdamW**, and **gradient clipping at 1.0**.
- **Generate** Amharic text from a prompt with **top-k + temperature sampling** and **causal masking**.
- **Log** training and validation loss curves and the learning-rate schedule.

## Architecture

| Component | File | Notes |
|---|---|---|
| Multi-head causal attention | `src/attention.py` | Q/K/V projections, scaled dot-product, causal mask via `register_buffer` |
| Feed-forward | `src/feedforward.py` | 4× expansion, GELU |
| Transformer block | `src/transformer_block.py` | Pre-norm + residual; norm → attn/FF → dropout → residual |
| Layer norm | `src/norm.py` | Custom impl with learnable scale & shift |
| Decoder-only model | `src/gpt_model.py` | Token + positional embeddings, N stacked blocks, untied output head |
| Loss | `src/loss.py` | Cross-entropy over flattened `(B*T, V)` logits + `evaluate_model` |
| Dataset | `src/dataset.py` | Sliding-window over the token stream with configurable stride |
| Cleaning | `src/utils_text.py`, `src/data_preprocessing.py` | Fidel, numerals, URLs/handles, whitespace |
| Training loop | `src/training.py` | Warmup + cosine decay, grad clip, end-of-epoch sample generation |
| CLI | `main.py` | `preprocess` / `train` / `generate` |

## Default model config

```python
{
    "vocab_size": 250002,    # XLM-RoBERTa tokenizer (overwritten at runtime)
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```
~124M parameters, GPT-2-small style.

## Install

```bash
git clone https://github.com/Abel-Marie/Large-Language-Model-For-Low-Resource-Languages.git
cd Large-Language-Model-For-Low-Resource-Languages
pip install -r requirements.txt
```

## How to run

### 1. Preprocess raw Amharic text

Place a raw Amharic `.txt` file (one document per line) under `data/raw/`, then:

```bash
python main.py preprocess \
  --input data/raw/amharic_corpus.txt \
  --output data/processed/cleaned_amh_data.txt \
  --verbose
```

### 2. Train

```bash
python main.py train --epochs 3 --batch-size 8 --lr 1e-4
```

Outputs:
- Checkpoint: `models/checkpoints/amharic_gpt.pt`
- Loss curve: `outputs/logs/loss_curve.png`

All training defaults live in `src/config.py` — override on the CLI as needed.

### 3. Generate

```bash
python main.py generate "ኢትዮጵያ በምሥራቅ አፍሪካ የምትገኝ" \
  --max-new-tokens 100 --temperature 0.8 --top-k 35
```

## Sample outputs

> 

## Hardware

> _

## What's next

- Train on a larger Amharic corpus.
- Switch to Rotary Position Embeddings (RoPE).
- Add evaluation: held-out perplexity and a small QA benchmark.
- DDP / FSDP for multi-GPU training.
- Quantized inference for on-device deployment.

## Project structure

```
.
├── main.py                  # CLI entry point
├── requirements.txt
├── LICENSE
├── README.md
├── src/
│   ├── attention.py
│   ├── config.py            # paths + model + training defaults
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── feedforward.py
│   ├── gpt_model.py
│   ├── loss.py
│   ├── norm.py
│   ├── training.py          # train_model, generate, encode/decode helpers
│   ├── transformer_block.py
│   └── utils_text.py        # Amharic-specific cleaning
├── scripts/
│   └── run_preprocessing.py # alternative preprocessing entry
├── data/{raw,interim,processed}/
├── models/{checkpoints,vocab}/
└── outputs/{logs,predictions}/
```

## License

[MIT](LICENSE) © 2025 Abel Marie Shiferaw

## Citation

```bibtex
@misc{shiferaw2026amharicgpt,
  author = {Abel Marie Shiferaw},
  title  = {Amharic GPT: A Low-Resource LLM from Scratch},
  year   = {2025},
  url    = {https://github.com/Abel-Marie/Large-Language-Model-For-Low-Resource-Languages}
}
```
