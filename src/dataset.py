import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, NUM_WORKERS


def build_transforms():
    train_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return train_tfms, eval_tfms


def build_dataloaders(data_dir, batch_size):
    train_tfms, eval_tfms = build_transforms()
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names
