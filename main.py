from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

DATA_DIR = Path("data/cats_and_dogs")
CLASS_NAMES = ["cat", "dog"]
IMG_SIZE = 224
BATCH_SIZE = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    return (image - mean) / std, label


def load_datasets(data_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_dataset, val_dataset


def apply_preprocessing(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)


def print_dataset_info(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
) -> None:
    class_names = train_dataset.class_names
    print(f"Classes: {class_names}")
    print(f"Class mapping: {dict(enumerate(class_names))}")
    print(f"Train batches: {train_dataset.cardinality().numpy()}")
    print(f"Val batches: {val_dataset.cardinality().numpy()}")


def download_data_if_needed(data_dir: Path) -> None:
    if any(data_dir.rglob("*.jpg")):
        return

    print("Downloading cats_vs_dogs dataset...")
    for split in ["train", "val"]:
        for class_name in CLASS_NAMES:
            (data_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    dataset = tfds.load("cats_vs_dogs", split="train", as_supervised=True)
    total = sum(1 for _ in dataset)
    train_cutoff = int(total * 0.8)

    dataset = tfds.load("cats_vs_dogs", split="train", as_supervised=True)
    for i, (image, label) in enumerate(dataset):
        split = "train" if i < train_cutoff else "val"
        class_name = CLASS_NAMES[label.numpy()]
        img_path = data_dir / split / class_name / f"{i}.jpg"
        tf.keras.utils.save_img(str(img_path), image.numpy())

    print("Download complete.")


if __name__ == "__main__":
    download_data_if_needed(DATA_DIR)
    train_dataset, val_dataset = load_datasets(DATA_DIR)
    print_dataset_info(train_dataset, val_dataset)

    train_dataset = apply_preprocessing(train_dataset)
    val_dataset = apply_preprocessing(val_dataset)

    images, labels = next(iter(train_dataset))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
