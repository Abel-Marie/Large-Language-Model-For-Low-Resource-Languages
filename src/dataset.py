import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        # Create sliding windows over the tokenized input
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            if len(input_chunk) == max_length and len(target_chunk) == max_length:
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(text, tokenizer, batch_size, max_length, stride, shuffle=True):
    """
    Create a PyTorch DataLoader from raw text for GPT-style training.

    Args:
        text (str): Raw input text.
        tokenizer: Tokenizer with an `.encode()` method.
        batch_size (int): Batch size.
        max_length (int): Length of each training sequence.
        stride (int): Overlap between windows.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader for training.
    """
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)