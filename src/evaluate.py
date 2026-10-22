# Test evaluation code extracted from the uploaded Colab notebook.
# Run the notebook for the complete dataset/model setup before using this script.

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

PROJECT_DIR = Path("/content/Deep-Residual-Image-Classification")
MODELS_DIR = PROJECT_DIR / "models"
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

for folder in [METRICS_DIR, FIGURES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 1. Load best checkpoint selected by validation loss
checkpoint_path = MODELS_DIR / "best_model.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Loaded best checkpoint from epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])

# 2. Evaluate on the untouched official test dataset
all_labels = []
all_predictions = []
all_probabilities = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):
            logits = model(images)

        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = probabilities.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.append(probabilities.cpu().numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.concatenate(all_probabilities, axis=0)

class_names = test_dataset.classes

# 3. Calculate metrics
test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
test_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
test_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)
test_roc_auc = roc_auc_score(
    y_true, y_prob, multi_class="ovr", average="macro"
)

# Multiclass average precision: one-vs-rest per class, then macro average
y_true_onehot = np.eye(len(class_names))[y_true]
test_average_precision = average_precision_score(
    y_true_onehot, y_prob, average="macro"
)

metrics = {
    "test_samples": int(len(y_true)),
    "accuracy": float(test_accuracy),
    "macro_precision": float(test_precision),
    "macro_recall": float(test_recall),
    "macro_f1": float(test_f1),
    "macro_roc_auc_ovr": float(test_roc_auc),
    "macro_average_precision": float(test_average_precision),
    "selected_checkpoint_epoch": int(checkpoint["epoch"]),
    "validation_loss_at_checkpoint": float(checkpoint["best_val_loss"])
}

print("\n" + "=" * 60)
print("OFFICIAL CIFAR-10 TEST RESULTS")
print("=" * 60)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)" if value <= 1 else f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

# 4. Per-class report
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\nPER-CLASS CLASSIFICATION REPORT")
print(report_text)

pd.DataFrame(report_dict).transpose().to_csv(
    METRICS_DIR / "classification_report.csv",
    index=True
)

# 5. Save numeric metrics and predictions
with open(METRICS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame({
    "true_label": y_true,
    "true_class": [class_names[i] for i in y_true],
    "predicted_label": y_pred,
    "predicted_class": [class_names[i] for i in y_pred],
    "correct": (y_true == y_pred),
    "confidence": y_prob.max(axis=1)
}).to_csv(
    PROJECT_DIR / "outputs" / "predictions" / "test_predictions.csv",
    index=False
)

# 6. Confusion matrix visualization
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 9))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)
ax.set_title("CIFAR-10 Test Set Confusion Matrix")
plt.tight_layout()

cm_path = FIGURES_DIR / "test_confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nSaved test metrics:", METRICS_DIR / "test_metrics.json")
print("Saved classification report:", METRICS_DIR / "classification_report.csv")
print("Saved predictions:", PROJECT_DIR / "outputs" / "predictions" / "test_predictions.csv")
print("Saved confusion matrix:", cm_path)
print("\nSUCCESS: Official test evaluation completed.")
