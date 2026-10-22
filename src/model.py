# Model architecture extracted from the uploaded Colab notebook.
# The notebook remains the canonical end-to-end workflow.

import torch
import torch.nn as nn
import json
from pathlib import Path

# Main project directory
PROJECT_DIR = Path("/content/Deep-Residual-Image-Classification")

class ResidualBlock(nn.Module):
    """Two convolution layers with an identity shortcut."""

    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Identity shortcut
        out = out + identity
        out = self.relu(out)

        # Downsample spatial dimensions
        out = self.pool(out)

        return out


class DeepResidualCNN(nn.Module):
    """CIFAR-10 residual CNN based on the project report."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.residual_block1 = ResidualBlock(64)
        self.residual_block2 = ResidualBlock(64)

        self.dropout = nn.Dropout2d(p=0.3)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# Initialize model on available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepResidualCNN(num_classes=10).to(device)

# Count trainable parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

# Test one batch with a forward pass
sample_images, sample_labels = next(iter(train_loader))
sample_images = sample_images.to(device)

model.eval()
with torch.no_grad():
    sample_outputs = model(sample_images)

print("=" * 55)
print("DEEP RESIDUAL CNN MODEL VERIFICATION")
print("=" * 55)
print("Device:", device)
print("Architecture:", model.__class__.__name__)
print("Trainable parameters:", f"{total_parameters:,}")
print("Input batch shape:", tuple(sample_images.shape))
print("Output logits shape:", tuple(sample_outputs.shape))
print("Expected classes:", 10)

assert sample_outputs.shape == (sample_images.shape[0], 10)
assert torch.isfinite(sample_outputs).all()

# Save architecture summary to project reports
summary = {
    "architecture": "DeepResidualCNN",
    "input_shape": [3, 32, 32],
    "num_classes": 10,
    "trainable_parameters": total_parameters,
    "device_used_for_test": str(device)
}

summary_path = PROJECT_DIR / "reports" / "model_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nModel summary saved:", summary_path)
print("SUCCESS: Forward pass completed.")
