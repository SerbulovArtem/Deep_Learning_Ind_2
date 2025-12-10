"""Test script for collate_fn to verify dynamic padding works correctly."""
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from models import get_tokenizer

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"


class TextDataset(Dataset):
    """Dataset that tokenizes texts without padding."""
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


def load_train_data(path: Path):
    texts = []
    labels = []
    label_names = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
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


def test_collate_fn():
    """Test the collate_fn with various batch sizes."""
    print("=" * 60)
    print("Testing collate_fn for dynamic padding")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = get_tokenizer("uklfr/gottbert-base")
    
    # Load data
    train_texts, train_label_indices, class_names = load_train_data(TRAIN_PATH)
    
    # Create dataset with full data to see real length distribution
    dataset = TextDataset(train_texts, train_label_indices, tokenizer, max_length=256)
    
    # Show length distribution before sorting
    print(f"\nDataset size: {len(dataset)}")
    print(f"Length distribution:")
    print(f"  Min: {min(dataset.lengths)}")
    print(f"  Max: {max(dataset.lengths)}")
    print(f"  Mean: {sum(dataset.lengths) / len(dataset.lengths):.1f}")
    print(f"  Sequences at max_length (256): {sum(1 for l in dataset.lengths if l == 256)}")
    print(f"  Sequences < 200: {sum(1 for l in dataset.lengths if l < 200)}")
    print(f"  Sequences < 150: {sum(1 for l in dataset.lengths if l < 150)}")
    print(f"  Sequences < 100: {sum(1 for l in dataset.lengths if l < 100)}")
    
    # Sort by length
    dataset.sort_by_length()
    
    print(f"\nAfter sorting (descending):")
    print(f"  First 10 lengths: {dataset.lengths[:10]}")
    print(f"  Last 10 lengths: {dataset.lengths[-10:]}")
    
    # Define collate function
    def collate_fn(batch):
        """Pads batch to the longest sequence in the batch."""
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
    
    # Test with different batch sizes - pick indices to show variety
    # Find indices with different length ranges
    total = len(dataset)
    test_cases = [
        (3, [0, 1, 2]),  # First 3 (longest)
        (5, [0, 1, 2, 3, 4]),  # First 5
        (3, [total//4, total//4+1, total//4+2]),  # 25% through
        (4, [total//2, total//2+1, total//2+2, total//2+3]),  # Middle
        (5, [total-10, total-9, total-8, total-7, total-6]),  # Near end (shortest)
    ]
    
    for batch_num, (batch_size, indices) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {batch_num}: Batch size {batch_size} ---")
        
        # Get batch
        test_batch = [dataset[i] for i in indices]
        
        # Show original lengths
        original_lengths = [len(item['input_ids']) for item in test_batch]
        print(f"Original input_ids lengths: {original_lengths}")
        print(f"Min length: {min(original_lengths)}, Max length: {max(original_lengths)}")
        
        # Apply collate_fn
        collated_data, labels = collate_fn(test_batch)
        
        # Show results
        print(f"Collated input_ids shape: {collated_data['input_ids'].shape}")
        print(f"Collated attention_mask shape: {collated_data['attention_mask'].shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Verify padding
        expected_max_len = max(original_lengths)
        actual_max_len = collated_data['input_ids'].shape[1]
        
        if expected_max_len == actual_max_len:
            print(f"✓ Padding correct: all sequences padded to {actual_max_len}")
        else:
            print(f"✗ ERROR: Expected {expected_max_len}, got {actual_max_len}")
        
        # Verify attention masks
        for i, orig_len in enumerate(original_lengths):
            mask = collated_data['attention_mask'][i]
            ones_count = mask.sum().item()
            if ones_count == orig_len:
                print(f"  ✓ Sample {i}: attention mask correct ({ones_count} ones)")
            else:
                print(f"  ✗ Sample {i}: ERROR - expected {orig_len} ones, got {ones_count}")
        
        # Show memory efficiency
        total_elements = collated_data['input_ids'].numel()
        actual_tokens = sum(original_lengths)
        padding_tokens = total_elements - actual_tokens
        efficiency = (actual_tokens / total_elements) * 100
        print(f"Efficiency: {efficiency:.1f}% ({actual_tokens}/{total_elements} tokens, {padding_tokens} padding)")
    
    # Test efficiency comparison
    print("\n" + "=" * 60)
    print("Efficiency Comparison: Dynamic vs Fixed Padding")
    print("=" * 60)
    
    # Simulate batch from sorted data
    batch_size = 16
    total = len(dataset)
    
    # Different batching strategies
    test_strategies = [
        ("Sorted - Longest sequences (0-15)", list(range(16))),
        ("Sorted - Mid-range sequences", list(range(total//2, total//2 + 16))),
        ("Sorted - Shortest sequences", list(range(total-16, total))),
        ("Random/Unsorted mix", [0, total//4, total//2, total-10, 5, total//4+5, total//2+5, total-20,
                                  10, total//4+10, total//2+10, total-30, 15, total//4+15, total//2+15, total-40]),
    ]
    
    for name, indices in test_strategies:
        batch = [dataset[i] for i in indices]
        lengths = [len(item['input_ids']) for item in batch]
        
        collated_data, _ = collate_fn(batch)
        
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = sum(lengths) / len(lengths)
        total_elements = collated_data['input_ids'].numel()
        actual_tokens = sum(lengths)
        efficiency = (actual_tokens / total_elements) * 100
        waste = total_elements - actual_tokens
        
        print(f"\n{name}:")
        print(f"  Lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}, range={max_len-min_len}")
        print(f"  Efficiency: {efficiency:.1f}% ({actual_tokens}/{total_elements} tokens)")
        print(f"  Wasted padding: {waste} tokens ({waste/total_elements*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_collate_fn()
