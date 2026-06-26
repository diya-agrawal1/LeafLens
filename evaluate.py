import argparse
import os

# Workaround for OpenMP duplicate runtime conflicts on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DEVICE
from src.dataset import build_dataloaders
from src.model import PlantDiseaseCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs("reports", exist_ok=True)

    _, _, test_loader, _ = build_dataloaders(args.data_dir, args.batch_size)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(cm, class_names, "reports/confusion_matrix.png")
    print(report)
    print("Saved evaluation report and confusion matrix.")


if __name__ == "__main__":
    main()
