import os
import pandas as pd

from src.utils_text import clean_amharic_text


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({'text': lines})


def clean_corpus(text_lines, min_length=10, verbose=False):
    cleaned = []
    for line in text_lines:
        c = clean_amharic_text(line)
        if len(c) >= min_length:
            cleaned.append(c)

    if verbose:
        print(f"Total lines: {len(text_lines)}")
        print(f"After cleaning: {len(cleaned)}")
        print(f"Total chars: {sum(len(x) for x in cleaned)}")
    return cleaned

def save_cleaned(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')