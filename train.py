import csv
from pathlib import Path
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
import mlflow

from models import GelectraClassifier, GottBERTClassifier, GermanBERTClassifier, get_tokenizer
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
TRAIN_PATH = DATA_DIR / "train.csv"  # Use full dataset (more data often helps)
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


def main(seed: int = 42, n_folds: int = 5):
    # Set seed for reproducibility
    set_seed(seed)
    
    mlflow.autolog(disable=True)
    mlflow.login()

    # NOTE: Here is the tokenizer

    # tokenizer = get_tokenizer("deepset/gelectra-base")  # GELECTRA base
    # tokenizer = get_tokenizer("uklfr/gottbert-base")  # GottBERT
    # tokenizer = get_tokenizer("deepset/gelectra-large")  # GELECTRA large
    tokenizer = get_tokenizer("bert-base-german-cased")  # German BERT

    # Load train data and build label mapping
    train_texts, train_label_indices, class_names = load_train_data(TRAIN_PATH)
    
    # Convert to numpy arrays for easier indexing
    train_texts = np.array(train_texts)
    train_label_indices = np.array(train_label_indices)

    # Setup k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    best_val_acc = 0.0
    best_fold = -1
    best_model_path = None
    
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_label_indices), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold}/{n_folds}")
        logger.info(f"{'='*50}")
        
        # Split data for this fold
        train_texts_split = train_texts[train_idx].tolist()
        val_texts_split = train_texts[val_idx].tolist()
        train_labels_split = train_label_indices[train_idx].tolist()
        val_labels_split = train_label_indices[val_idx].tolist()
        
        max_length = 256  # Capture more context from longer German articles
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

        batch_size=32  # Larger batch = more stable gradients

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
        # model = GelectraClassifier(model_name="deepset/gelectra-large", num_labels=len(class_names)).to(device)
        model = GermanBERTClassifier(model_name="bert-base-german-cased", num_labels=len(class_names)).to(device)

        # Setup optimizer and scheduler
        weight_decay = 0.01  # Standard weight decay
        start_lr = 2e-5  # Standard BERT learning rate
        optimizer = AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay)
        
        # Warmup + linear decay scheduler (6% warmup - standard for BERT)
        epochs = 5  # Standard 3-5 epochs for BERT
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
            "max_length": max_length,
            "fold": fold,
            "n_folds": n_folds,
        }

        trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, compile=False)
        trainer.fit(train_loader, val_loader, epochs=epochs, params=params)
        
        # Get final validation accuracy
        val_acc = trainer.best_val_acc if hasattr(trainer, 'best_val_acc') else 0.0
        fold_results.append({"fold": fold, "val_acc": val_acc})
        
        # Save model for this fold
        out_dir = Path("models")
        out_dir.mkdir(exist_ok=True)
        fold_save_path = f"{out_dir}/{model._get_name()}_seed{seed}_fold{fold}.pth"
        trainer.save(fold_save_path, include_optimizer=False, include_scheduler=False)
        
        # Track best fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_fold = fold
            best_model_path = fold_save_path
        
        logger.info(f"Fold {fold} validation accuracy: {val_acc:.4f}")
    
    # Print cross-validation summary
    logger.info(f"\n{'='*50}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*50}")
    for result in fold_results:
        logger.info(f"Fold {result['fold']}: {result['val_acc']:.4f}")
    
    avg_val_acc = np.mean([r['val_acc'] for r in fold_results])
    std_val_acc = np.std([r['val_acc'] for r in fold_results])
    logger.info(f"\nAverage validation accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    logger.info(f"Best fold: {best_fold} with accuracy: {best_val_acc:.4f}")
    logger.info(f"Best model saved at: {best_model_path}")
    
    # Create submission using the best fold's model
    logger.info(f"\nCreating submission with best model (fold {best_fold})...")
    
    # Reload best model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GermanBERTClassifier(model_name="bert-base-german-cased", num_labels=len(class_names)).to(device)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_texts, test_ids = load_test_data(TEST_PATH)
    test_dataset = TextTestDataset(test_texts, test_ids, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    trainer = Trainer(model, optimizer=None, scheduler=None, compile=False)
    submission_path = f"data/submission_seed{seed}_cv.csv"
    trainer.create_submission(test_loader, class_names, submission_path=submission_path)
    
    return best_model_path, submission_path, fold_results


if __name__ == "__main__":
    main()