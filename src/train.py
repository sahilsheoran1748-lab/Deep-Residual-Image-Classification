# Training loop extracted from the uploaded Colab notebook.
# Run the notebook for the complete dataset setup and variable initialization.

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

PROJECT_DIR = Path("/content/Deep-Residual-Image-Classification")
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 15
MAX_GRAD_NORM = 1.0
USE_AMP = torch.cuda.is_available()

# Use the previously configured model, optimizer, scheduler,
# criterion and scaler from Step 11.
history = []
best_val_loss = float("inf")
best_epoch = 0
start_time = time.time()

metrics_path = METRICS_DIR / "training_history.csv"
checkpoint_path = MODELS_DIR / "best_model.pth"
last_checkpoint_path = MODELS_DIR / "last_checkpoint.pth"

def run_validation():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples

try:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=True
        )

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM
            )

            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

            progress.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                accuracy=f"{running_correct / total_samples:.3f}"
            )

        train_loss = running_loss / total_samples
        train_accuracy = running_correct / total_samples

        val_loss, val_accuracy = run_validation()

        # Reduce learning rate when validation loss plateaus
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds
        }
        history.append(record)

        # Save history after every epoch
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)

        # Save the best checkpoint using validation loss only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_accuracy": val_accuracy,
                    "class_names": train_dataset.classes,
                    "config": training_config
                },
                checkpoint_path
            )

        # Save a recovery checkpoint after every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            },
            last_checkpoint_path
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_seconds:.1f}s"
        )

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Completed epochs have been saved.")

total_minutes = (time.time() - start_time) / 60

# Save a summary of the training run
training_summary = {
    "epochs_completed": len(history),
    "best_epoch": best_epoch,
    "best_validation_loss": (
        best_val_loss if best_epoch > 0 else None
    ),
    "training_minutes": round(total_minutes, 2),
    "best_checkpoint": str(checkpoint_path),
    "metrics_csv": str(metrics_path)
}

summary_path = LOGS_DIR / "training_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(training_summary, f, indent=4)

print("\n" + "=" * 60)
print("TRAINING RUN SUMMARY")
print("=" * 60)
print(json.dumps(training_summary, indent=2))
print("\nTraining history:", metrics_path)
print("Best model checkpoint:", checkpoint_path)
print("Latest recovery checkpoint:", last_checkpoint_path)
