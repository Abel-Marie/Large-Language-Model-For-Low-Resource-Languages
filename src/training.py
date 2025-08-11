from src.dataset import create_dataloader
from src.gpt_model import GPTModel
from loss.py import calc_loss_batch, calc_loss_loader, evaluate_model
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import XLMRobertaTokenizerFast
import sys
import math
import matplotlib.pyplot as plt


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_loader = create_dataloader(
    text_data[:split_idx],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    text_data[split_idx:],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    context_size = model.cfg["context_length"]
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < top_logits[:, [-1]], -torch.inf, logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def train_model(model, train_loader, val_loader, optimizer, device, n_epochs,
                eval_freq, eval_iter, warmup_steps, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    model.to(device)
    peak_lr = optimizer.param_groups[0]['lr']
    total_training_steps = len(train_loader) * n_epochs

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1
            if global_step < warmup_steps:
                lr = peak_lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                lr = peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step > 0 and global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss); val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen); track_lrs.append(lr)
                print(f"Epoch {epoch+1:02d}/{n_epochs:02d} | Step {global_step:06d} | "
                      f"LR {lr:.6f} | Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f}")

        model.eval()
        print("\n--- Generating Sample Text ---")
        encoded_context = text_to_token_ids(start_context, tokenizer).to(device)
        generated_ids = generate(model, encoded_context, max_new_tokens=50, top_k=25, temperature=0.7)
        print(f">>> {token_ids_to_text(generated_ids, tokenizer)}\n")

    return train_losses, val_losses, track_tokens_seen, track_lrs



NUM_EPOCHS = 3
BATCH_SIZE = 8 
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 0.1
EVAL_FREQ = 100
EVAL_ITER = 20
WARMUP_STEPS = 20


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {str(device).upper()}")


# --- File Path ---
filepath = "/data/processed/cleaned_amh_data.txt"


# --- Load Raw Text ---
try:
    with open(filepath, "r", encoding="utf-8", errors='replace') as f:
        text_data = f.read()
    print(f"File loaded successfully. Total characters: {len(text_data)}")
except FileNotFoundError:
    print(f"❌ File not found: {filepath}. Please upload 'cleaned_amh_data.txt'.")
    sys.exit(1)


# --- Initialize Tokenizer ---
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
GPT_CONFIG_124M["vocab_size"] = tokenizer.vocab_size # Dynamically set vocab size
print(f"Tokenizer loaded. Vocabulary size: {GPT_CONFIG_124M['vocab_size']}")


# Execute Trainig

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
print(f" Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(" Starting model training...")
train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, optimizer, device,
    n_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
    warmup_steps=WARMUP_STEPS, start_context="ሰላም! ይህ የ", tokenizer=tokenizer
)


# Plot Results

if train_losses: # Only plot if training happened
    print("\nTraining finished. Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(tokens_seen, train_losses, label="Training Loss")
    ax1.plot(tokens_seen, val_losses, linestyle="--", label="Validation Loss")
    ax1.set_ylabel("Loss"); ax1.set_title("Training and Validation Losses")
    ax1.legend(); ax1.grid(True)
    ax2.plot(tokens_seen, lrs, label="Learning Rate", color='green')
    ax2.set_xlabel("Tokens Seen"); ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()


# generating text 

print("\n🤖 Generating final sample text...")
model.eval()
start_context = "ኢትዮጵያ በምሥራቅ አፍሪካ የምትገኝ" 
encoded_context = text_to_token_ids(start_context, tokenizer).to(device)
generated_ids = generate(model, encoded_context, max_new_tokens=100, top_k=35, temperature=0.8)
print("\n--- Generated Text ---")
print(token_ids_to_text(generated_ids, tokenizer))
print("------------------------\n")
