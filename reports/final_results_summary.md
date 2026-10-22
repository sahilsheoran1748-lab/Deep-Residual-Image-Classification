# Deep Residual CNN — Final Results Summary

## 1. Project Overview

This project applies a Deep Residual Convolutional Neural Network (CNN) to image classification on the CIFAR-10 dataset using PyTorch.

The dataset contains 10 image classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## 2. Model Details

- **Architecture:** Deep Residual CNN
- **Dataset:** CIFAR-10
- **Training samples:** 45,000
- **Validation samples:** 5,000
- **Official test samples:** 10,000
- **Optimizer:** AdamW
- **Initial learning rate:** 0.001
- **Batch size:** 64
- **Training epochs:** 15

## 3. Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 77.90% |
| Macro Precision | 77.93% |
| Macro Recall | 77.90% |
| Macro F1-Score | 77.71% |
| Macro ROC-AUC (OVR) | 97.47% |
| Macro Average Precision | 86.15% |

## 4. Per-Class F1-Score

| Class | F1-Score |
|---|---:|
| Airplane | 79.70% |
| Automobile | 90.05% |
| Bird | 67.36% |
| Cat | 55.94% |
| Deer | 75.58% |
| Dog | 69.36% |
| Frog | 82.12% |
| Horse | 81.43% |
| Ship | 88.07% |
| Truck | 87.51% |

## 5. Observations

- Overall test accuracy was 77.90%.
- Automobile achieved the highest per-class F1-score.
- Cat had the lowest per-class F1-score.
- Cat and dog were frequently confused by the model.
- Further improvements could focus on data augmentation, model tuning, and distinguishing visually similar classes.

## 6. Conclusion

The trained Deep Residual CNN demonstrated useful performance on CIFAR-10 image classification. The evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, and per-class results.

## 7. Reproducibility

Refer to the project notebook and training configuration for implementation details. Reported metrics correspond to the recorded evaluation results for this experiment.
