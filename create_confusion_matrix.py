import os
import csv
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.cluster.hierarchy import linkage, leaves_list
from torch.utils.data import DataLoader

from models import GottBERTClassifier, get_tokenizer
from train import TextDataset

MODEL_FILE = "models/GottBERTClassifier_final.pth"
DATA_PATH = Path("data/train_filtered.csv")  # Use filtered data or "data/train.csv"
BATCH = 64
OUT_DIR = "data/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

print(f"Loaded {len(texts)} samples with {len(unique_labels)} classes")
print(f"Classes: {unique_labels}")

# Create dataset and dataloader
dataset = TextDataset(texts, label_indices, tokenizer, max_length=128)

def collate_fn(batch):
    """Dynamic padding per batch."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["label"] for item in batch]
    
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
    }, torch.tensor(labels, dtype=torch.long)

loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=collate_fn, num_workers=1)

# Load model
model = GottBERTClassifier(model_name="uklfr/gottbert-base", num_labels=len(unique_labels))

state = torch.load(MODEL_FILE, map_location=device)

if isinstance(state, dict) and "model_state" in state:
    model_state = state["model_state"]
    missing, unexpected = model.load_state_dict(model_state, strict=True)
    print(f"Loaded model state. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
else:
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded model state. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

model.to(device).eval()

# Get predictions
preds, targets = [], []
with torch.inference_mode():
    for xb, yb in loader:
        xb = {k: v.to(device) for k, v in xb.items()}
        logits = model(xb)
        p = logits.argmax(1).cpu().numpy()
        preds.append(p)
        targets.append(yb.numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

# Confusion matrix
cm = confusion_matrix(targets, preds)
cm_row = cm / (cm.sum(1, keepdims=True) + 1e-12)

classes = unique_labels

# Per-class metrics
prec, rec, f1, support = precision_recall_fscore_support(
    targets, preds, labels=range(len(classes)), zero_division=0
)
metrics = np.stack([support, prec, rec, f1], axis=1)

with open(f"{OUT_DIR}/per_class_metrics.csv", "w") as f:
    f.write("class,support,precision,recall,f1\n")
    for i, cls in enumerate(classes):
        f.write(f"{cls},{support[i]},{prec[i]:.4f},{rec[i]:.4f},{f1[i]:.4f}\n")

print(f"Saved per-class metrics to {OUT_DIR}/per_class_metrics.csv")

# Top confusing pairs (exclude diagonal)
off = cm.copy()
np.fill_diagonal(off, 0)
pairs = []
for i in range(off.shape[0]):
    for j in range(off.shape[1]):
        if off[i, j] > 0:
            pairs.append((off[i, j], i, j))
pairs.sort(reverse=True)
top_k = pairs[:25]

with open(f"{OUT_DIR}/top_confusions.txt", "w") as f:
    for cnt, i, j in top_k:
        f.write(f"{classes[i]} -> {classes[j]}: {int(cnt)}\n")

print(f"Saved top confusions to {OUT_DIR}/top_confusions.txt")

# Plot per-class recall sorted
order_rec = np.argsort(rec)
plt.figure(figsize=(10, 6))
plt.bar(range(len(classes)), rec[order_rec])
plt.xticks(range(len(classes)), [classes[i] for i in order_rec], rotation=45, ha='right')
plt.ylabel("Recall")
plt.title("Per-class Recall (sorted)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_recall.png", dpi=180)
plt.close()

print(f"Saved recall plot to {OUT_DIR}/per_class_recall.png")

# Full confusion matrix (row-normalized)
plt.figure(figsize=(10, 8))
plt.imshow(cm_row, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45, ha='right', fontsize=8)
plt.yticks(range(len(classes)), classes, fontsize=8)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (row-normalized)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_full.png", dpi=180)
plt.close()

print(f"Saved full confusion matrix to {OUT_DIR}/confusion_matrix_full.png")

# Clustered confusion matrix
profiles = cm_row
Z = linkage(profiles, method="average", metric="euclidean")
leaf_order = leaves_list(Z)
cluster_cm = cm_row[leaf_order][:, leaf_order]
cluster_classes = [classes[i] for i in leaf_order]

plt.figure(figsize=(10, 8))
plt.imshow(cluster_cm, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(cluster_classes)), cluster_classes, rotation=45, ha='right', fontsize=8)
plt.yticks(range(len(cluster_classes)), cluster_classes, fontsize=8)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Clustered Confusion Matrix (row-normalized)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_clustered.png", dpi=180)
plt.close()

print(f"Saved clustered confusion matrix to {OUT_DIR}/confusion_clustered.png")

# Summary statistics
acc = (preds == targets).mean()
print(f"\nOverall Accuracy: {acc:.4f}")
print(f"Mean Precision: {prec.mean():.4f}")
print(f"Mean Recall: {rec.mean():.4f}")
print(f"Mean F1: {f1.mean():.4f}")

print(f"\nAll artifacts written to {OUT_DIR}/")
