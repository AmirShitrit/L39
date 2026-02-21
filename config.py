from pathlib import Path

DATA_DIR = Path("data/cats_and_dogs")
CLASS_NAMES = ["cat", "dog"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
