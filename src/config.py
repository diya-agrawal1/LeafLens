import torch

IMAGE_SIZE = 128
NUM_WORKERS = 0
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
