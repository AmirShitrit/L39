from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_DIR = Path("data/cats_and_dogs")
CLASS_NAMES = ["cat", "dog"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

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


LAYER_COLORS = {
    "Conv2D": "#4C72B0",
    "MaxPooling2D": "#DD8452",
    "Flatten": "#55A868",
    "Dense": "#C44E52",
}

def save_architecture_image(model: tf.keras.Model, output_path: str = "architecture.png") -> None:
    layers = [l for l in model.layers if l.__class__.__name__ in LAYER_COLORS]
    n = len(layers)
    fig, ax = plt.subplots(figsize=(max(14, n * 1.4), 5))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F8F8")

    box_w, box_h, y0 = 0.75, 0.45, 0.275
    for i, layer in enumerate(layers):
        cls = layer.__class__.__name__
        color = LAYER_COLORS[cls]
        x = i + 0.5 - box_w / 2
        rect = mpatches.FancyBboxPatch(
            (x, y0), box_w, box_h,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor="white", facecolor=color, alpha=0.88,
        )
        ax.add_patch(rect)
        cx = i + 0.5
        output_shape = str(tuple(layer.output.shape))[1:-1]
        ax.text(cx, y0 + box_h + 0.06, cls, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=color)
        ax.text(cx, y0 + box_h / 2, output_shape, ha="center", va="center",
                fontsize=6.5, color="white", wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(i + 1 + 0.5 - box_w / 2, y0 + box_h / 2),
                        xytext=(x + box_w, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    legend_patches = [mpatches.Patch(color=c, label=k) for k, c in LAYER_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower center", ncol=4,
              framealpha=0.7, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    ax.set_title("CNN Architecture", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Architecture image saved to {output_path}")


def build_model() -> tf.keras.Model:
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])


def compile_model(model: tf.keras.Model) -> None:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
) -> tf.keras.callbacks.History:
    return model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
    )


if __name__ == "__main__":
    download_data_if_needed(DATA_DIR)
    train_dataset, val_dataset = load_datasets(DATA_DIR)
    print_dataset_info(train_dataset, val_dataset)

    train_dataset = apply_preprocessing(train_dataset)
    val_dataset = apply_preprocessing(val_dataset)

    images, labels = next(iter(train_dataset))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")

    model = build_model()
    model.summary()
    save_architecture_image(model)

    compile_model(model)
    train_model(model, train_dataset, val_dataset)
