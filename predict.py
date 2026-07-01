import argparse
import os

# Workaround for OpenMP duplicate runtime conflicts on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image
from torchvision import transforms

from src.config import DEVICE, IMAGE_SIZE
from src.model import PlantDiseaseCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

    print(f"Predicted class: {class_names[pred_idx]}")
    print(f"Confidence: {probs[pred_idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
