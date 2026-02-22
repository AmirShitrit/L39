import tensorflow as tf

from config import DATA_DIR
from data import apply_preprocessing, download_data_if_needed, load_datasets, print_dataset_info
from model import build_model, compile_model, train_model
from visualization import (
    plot_accuracy_vs_loss,
    plot_confusion_matrix,
    plot_training_history,
    save_architecture_image,
)

if __name__ == "__main__":
    tf.random.set_seed(42)

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
    history = train_model(model, train_dataset, val_dataset)
    plot_training_history(history)
    plot_confusion_matrix(model, val_dataset)
    plot_accuracy_vs_loss(history)
