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

from domainbed.lib.fast_vit import HuggingFaceFastViTBackbone
#
# ---------- Configuration ----------
CHECKPOINT_PATH = 'output_fastViT/8c2c2868a4054834ec4e8c992489e050/IID_best.pkl'
NUM_CLASSES = 5
DATA_DIR = 'data/DR/messidor_2'
OUTPUT_JSON = 'Test_Results/multi_domain/fastvit/test_env_3_results_1.json'
BATCH_SIZE = 16

print("Starting evaluation on dataset:", DATA_DIR)

# ---------- Load Model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HuggingFaceFastViTBackbone(num_classes=NUM_CLASSES)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
raw_state_dict = checkpoint["model_dict"]

# Remove "network." prefix if present
clean_state_dict = {
    k.replace("network.", ""): v for k, v in raw_state_dict.items()
}

missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
model.to(device)
model.eval()

print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

print(f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

# ---------- Dataset ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

dataset = ImageFolder(DATA_DIR, transform=transform)

expected_class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

dataset.class_to_idx = expected_class_to_idx
print("Enforced class_to_idx mapping:", dataset.class_to_idx)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Evaluation ----------
all_preds, all_labels, times = [], [], []

try:
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)

            # Handle tuple/list output from model
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

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
print(classification_report(all_labels, all_preds, labels=[0, 1, 2, 3, 4], target_names=['0', '1', '2', '3', '4'], zero_division=0))

# ---------- Save Results ----------
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=4)

print("\nEvaluation complete. Results saved to:", OUTPUT_JSON)

