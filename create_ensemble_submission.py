"""Create ensemble submission using majority voting from existing submission files."""
import csv
import glob
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")

# Find all seed submission files
submission_files = sorted(glob.glob(str(DATA_DIR / "submission_seed*.csv")))

if not submission_files:
    print("No seed submissions found! Train them first with train_ensemble.py")
    exit(1)

print(f"Found {len(submission_files)} submissions to ensemble:")
for f in submission_files:
    print(f"  - {Path(f).name}")

# Load all submissions
all_predictions = {}  # {id: [label1, label2, ...]}

for sub_file in submission_files:
    with open(sub_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row['id']
            label = row['label']
            if sample_id not in all_predictions:
                all_predictions[sample_id] = []
            all_predictions[sample_id].append(label)

print(f"\nProcessing {len(all_predictions)} test samples...")

# Perform majority voting
ensemble_predictions = []
for sample_id in all_predictions:
    labels = all_predictions[sample_id]
    # Count votes for each label
    vote_counts = Counter(labels)
    # Pick the most common label (majority vote)
    majority_label = vote_counts.most_common(1)[0][0]
    ensemble_predictions.append((sample_id, majority_label))

# Create submission
submission_path = "data/submission_ensemble.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    writer.writerows(ensemble_predictions)

print(f"\nEnsemble submission saved to {submission_path}")
