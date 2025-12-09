import numpy as np
from cleanlab.filter import find_label_issues
import logging
import torch
import csv
from pathlib import Path
from torch.utils.data import DataLoader

from models import GottBERTClassifier, get_tokenizer
from train import TextDataset

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
)

logger.info("Loading saved model for Cleanlab analysis")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILE = "models/GottBERTClassifier_final.pth"
DATA_PATH = Path("data/train.csv")  # Use original unfiltered data

# Load tokenizer
tokenizer = get_tokenizer("uklfr/gottbert-base")

# Load data
texts = []
labels = []
label_names = []
with DATA_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["label"].strip()
        text = row["text"].strip()
        texts.append(text)
        label_names.append(label)

# Create label mapping
unique_labels = sorted(set(label_names))
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
label_indices = [label_to_idx[lbl] for lbl in label_names]

logger.info(f"Loaded {len(texts)} samples with {len(unique_labels)} classes")

# Create dataset and dataloader
dataset = TextDataset(texts, label_indices, tokenizer, max_length=128)

def collate_fn(batch):
    """Dynamic padding per batch."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels_batch = [item["label"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        padding_length = max_len - len(ids)
        padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_length)
        padded_attention_masks.append(mask + [0] * padding_length)
    
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
    }, torch.tensor(labels_batch, dtype=torch.long)

eval_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1,
    pin_memory=torch.cuda.is_available()
)

# Load model
model = GottBERTClassifier(model_name="uklfr/gottbert-base", num_labels=len(unique_labels))
state_dict = torch.load(MODEL_FILE, map_location=device)

# Handle trainer save format
if isinstance(state_dict, dict) and "model_state" in state_dict:
    logger.info(f"Loaded container keys: {list(state_dict.keys())}")
    state_dict = state_dict["model_state"]

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    logger.warning(f"Missing keys after load: {missing}")
if unexpected:
    logger.warning(f"Unexpected keys ignored: {unexpected}")

model.to(device).eval()

# Get predictions
num_classes = len(unique_labels)
pred_probs = np.zeros((len(dataset), num_classes), dtype=np.float32)

logger.info("Computing predictions for all samples...")
with torch.inference_mode():
    idx = 0
    for xb, _ in eval_loader:
        xb = {k: v.to(device) for k, v in xb.items()}
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        bsz = probs.shape[0]
        pred_probs[idx:idx+bsz] = probs
        idx += bsz

labels_array = np.array(label_indices)

# Find label issues using Cleanlab
logger.info("Running Cleanlab to find label issues...")
issue_indices = find_label_issues(
    labels=labels_array,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence"
)

logger.info(f"Potential label issues found: {len(issue_indices)}")

# Derive suggested labels from model predictions
predicted_labels = pred_probs.argmax(axis=1)
predicted_probs = pred_probs.max(axis=1)
given_probs = pred_probs[np.arange(len(dataset)), labels_array]  # self-confidence

# Save issues with suggestions
issues_csv = "data/label_issues.csv"
with open(issues_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "index",
        "text",
        "given_label_id",
        "given_label",
        "predicted_label_id",
        "predicted_label",
        "predicted_prob",
        "self_confidence_given_label"
    ])
    for i in issue_indices:
        text = texts[i]
        given_label_id = label_indices[i]
        given_label = unique_labels[given_label_id]
        pred_lbl = int(predicted_labels[i])
        pred_label = unique_labels[pred_lbl]
        
        writer.writerow([
            i,
            text,  # Truncate long texts for CSV readability
            given_label_id,
            given_label,
            pred_lbl,
            pred_label,
            float(predicted_probs[i]),
            float(given_probs[i]),
        ])

logger.info(f"Label issues written to {issues_csv}")
logger.info(f"Top 10 issues (lowest self-confidence):")
for idx, i in enumerate(issue_indices[:10]):
    logger.info(
        f"  {idx+1}. Index={i} given={unique_labels[label_indices[i]]} → "
        f"predicted={unique_labels[predicted_labels[i]]} "
        f"(pred_prob={predicted_probs[i]:.4f}, self_conf={given_probs[i]:.4f})"
    )
