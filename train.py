import csv
from pathlib import Path
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
import mlflow

from models import GelectraClassifier, GottBERTClassifier, get_tokenizer
from trainer import Trainer

from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


class TextDataset(Dataset):
    """Dataset that tokenizes texts without padding.
    
    Sorting and dynamic padding will be handled by the DataLoader collate_fn.
    """
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels
        
        # Pre-tokenize without padding to get lengths for sorting
        self.encodings = [
            tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            for text in texts
        ]
        
        # Calculate lengths for sorting
        self.lengths = [len(enc["input_ids"]) for enc in self.encodings]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return raw encoding dict (no padding yet) and label
        encoding = self.encodings[idx]
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": self.labels[idx],
        }
    
    def sort_by_length(self):
        """Sort dataset by sequence length (descending) for efficient batching."""
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i], reverse=True)
        self.texts = [self.texts[i] for i in sorted_indices]
        self.labels = [self.labels[i] for i in sorted_indices]
        self.encodings = [self.encodings[i] for i in sorted_indices]
        self.lengths = [self.lengths[i] for i in sorted_indices]


class TextTestDataset(Dataset):
    def __init__(self, texts, ids, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        xb = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        sample_id = self.ids[idx]
        return xb, sample_id


def load_train_data(path: Path):
    texts = []
    labels = []
    label_names = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            text = row["text"].strip()
            texts.append(text)
            labels.append(label)
            label_names.append(label)

    # Stable mapping label string -> index
    unique_labels = sorted(set(label_names))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_labels = [label_to_idx[lbl] for lbl in labels]
    return texts, idx_labels, unique_labels


def load_test_data(path: Path):
    texts = []
    ids = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"].strip())
            texts.append(row["text"].strip())
    return texts, ids


class BucketBatchSampler:
    """Sampler that creates batches of similar-length sequences, then shuffles batch order.
    
    This provides both efficiency (similar lengths = less padding) and randomness
    (different batch order each epoch) for better generalization.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Dataset is already sorted by length
        # Create batches of consecutive samples (similar lengths)
        self.batches = []
        for i in range(0, len(dataset), batch_size):
            self.batches.append(list(range(i, min(i + batch_size, len(dataset)))))
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)  # Shuffle batch order
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    mlflow.autolog(disable=True)
    mlflow.login()

    # NOTE: Here is the tokenizer

    # tokenizer = get_tokenizer("deepset/gelectra-base")  # deepset/gelectra-base
    # tokenizer = get_tokenizer("uklfr/gottbert-base")  # GottBERT
    tokenizer = get_tokenizer("deepset/gelectra-large")

    # Load train data and build label mapping
    train_texts, train_label_indices, class_names = load_train_data(TRAIN_PATH)

    # Stratified train/val split (e.g. 90%/10%)
    train_texts_split, val_texts_split, train_labels_split, val_labels_split = train_test_split(
        train_texts,
        train_label_indices,
        test_size=0.1,
        stratify=train_label_indices,
        random_state=42,
    )
    max_length = 128
    train_dataset = TextDataset(train_texts_split, train_labels_split, tokenizer, max_length=max_length)
    val_dataset = TextDataset(val_texts_split, val_labels_split, tokenizer, max_length=max_length)
    
    # Sort datasets by length for efficient batching
    train_dataset.sort_by_length()
    val_dataset.sort_by_length()
    
    # Custom collate function for dynamic padding per batch
    def collate_fn(batch):
        """Pads batch to the longest sequence in the batch."""
        # Extract input_ids, attention_mask, and labels
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Find max length in this batch
        max_len = max(len(ids) for ids in input_ids)
        
        # Pad to max_len
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_length)
            padded_attention_masks.append(mask + [0] * padding_length)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        }, torch.tensor(labels, dtype=torch.long)

    batch_size=16

    # Create batch sampler that shuffles batches (not individual samples)
    train_batch_sampler = BucketBatchSampler(train_dataset, batch_size, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,  # Use batch_sampler for efficient + random batching
        collate_fn=collate_fn,
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )

    # NOTE: Here is the model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = GelectraClassifier(model_name="deepset/gelectra-base", num_labels=len(class_names)).to(device)
    # model = GottBERTClassifier(model_name="uklfr/gottbert-base", num_labels=len(class_names)).to(device)
    model = GelectraClassifier(model_name="deepset/gelectra-large", num_labels=len(class_names)).to(device)

    # Setup optimizer and scheduler
    weight_decay = 0.2
    start_lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay) # change lr
    
    # Warmup + linear decay scheduler (warmup 6%)
    epochs = 8
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.06 * total_steps)  # 6% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    params = {
        "weight_decay": weight_decay,
        "start_lr": start_lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "max_length": max_length
    }

    trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, compile=False)
    trainer.fit(train_loader, val_loader, epochs=epochs, params=params)

    # Save the model
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    save_path = f"{out_dir}/{model._get_name()}_final.pth"
    trainer.save(save_path, include_optimizer=False, include_scheduler=False)

    # Optional: create submission on test set
    test_texts, test_ids = load_test_data(TEST_PATH)
    test_dataset = TextTestDataset(test_texts, test_ids, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    trainer.create_submission(test_loader, class_names, submission_path="data/submission.csv")


if __name__ == "__main__":
    main()