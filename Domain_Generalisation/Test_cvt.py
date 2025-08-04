import os
import time
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from collections import Counter

# DomainBed algorithm loader
from domainbed.algorithms import get_algorithm_class

# ---------- Configuration ----------
CHECKPOINT_PATH = 'output_cvt13/f873a67f75338d04f234a40e176ee632/IID_best.pkl'
NUM_CLASSES = 5
DATA_DIR = 'data/DR/messidor'
OUTPUT_JSON = 'Test_Results/multi_domain/cvt/test_env_2_results_1.json'
BATCH_SIZE = 16

print("Starting evaluation on dataset:", DATA_DIR)

# ---------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Load Checkpoint ----------
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# ---------- Rebuild Algorithm ----------
algorithm_class = get_algorithm_class(checkpoint["args"]["algorithm"])
algorithm = algorithm_class(
    input_shape=checkpoint["model_input_shape"],
    num_classes=checkpoint["model_num_classes"],
    num_domains=checkpoint["model_num_domains"],
    hparams=checkpoint["model_hparams"]
)
algorithm.load_state_dict(checkpoint["model_dict"], strict=True)
algorithm.to(device)
algorithm.eval()

# ---------- Extract CvT Model ----------
model = algorithm.network
model.eval()

print(f"Model loaded. Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ---------- Dataset ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

dataset = ImageFolder(DATA_DIR, transform=transform)
expected_classes = ['0', '1', '2', '3', '4']
actual_classes = dataset.classes

# Set class_to_idx explicitly
dataset.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(expected_classes)}
print("Enforced class_to_idx:", dataset.class_to_idx)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Evaluation ----------
all_preds, all_labels, times = [], [], []

try:
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            start = time.time()
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]  # Final classifier output
            end = time.time()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            times.extend([(end - start) / images.size(0)] * images.size(0))

except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)

# ---------- Metrics ----------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
avg_time = sum(times) / len(times)

results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'avg_inference_time_per_sample_sec': avg_time,
    'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
}

# ---------- Print Summary ----------
print("\nEvaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=[str(i) for i in range(NUM_CLASSES)],
    zero_division=0
))

# ---------- Save Results ----------
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=4)

print("\nEvaluation complete. Results saved to:", OUTPUT_JSON)
print("Test preds:", Counter(all_preds))
print("Test labels:", Counter(all_labels))
