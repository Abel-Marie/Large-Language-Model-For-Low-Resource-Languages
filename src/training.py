"""Training utilities for the Amharic GPT model.

Pure functions — no top-level execution. Drive training from `main.py train`.
"""
import math

import torch

from src.loss import calc_loss_batch, evaluate_model


def text_to_token_ids(text, tokenizer):
    """Encode text -> long tensor of shape [1, T] (with batch dim)."""
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """Decode a [1, T] (or [T]) long tensor back to text."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Autoregressive sampling with optional top-k filtering and temperature."""
    context_size = model.cfg["context_length"]
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < top_logits[:, [-1]], float("-inf"), logits
            )

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    n_epochs,
    eval_freq,
    eval_iter,
    warmup_steps,
    start_context,
    tokenizer,
):
    """Train with linear warmup -> cosine decay LR, AdamW, grad clip @ 1.0."""
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    model.to(device)
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:
                lr = peak_lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(
                    1, (total_training_steps - warmup_steps)
                )
                lr = peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step > 0 and global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                track_lrs.append(lr)
                print(
                    f"Epoch {epoch + 1:02d}/{n_epochs:02d} | "
                    f"Step {global_step:06d} | LR {lr:.6f} | "
                    f"Train {train_loss:.3f} | Val {val_loss:.3f}"
                )

        encoded_context = text_to_token_ids(start_context, tokenizer).to(device)
        generated_ids = generate(
            model, encoded_context, max_new_tokens=50, top_k=25, temperature=0.7
        )
        print(
            f"\n--- Sample after epoch {epoch + 1} ---\n"
            f">>> {token_ids_to_text(generated_ids, tokenizer)}\n"
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs
