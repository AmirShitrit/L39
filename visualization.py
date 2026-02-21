import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES

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


def plot_training_history(
    history: tf.keras.callbacks.History,
    output_path: str = "training_history.png",
) -> None:
    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    ax_acc.plot(epochs, history.history["accuracy"], label="Train")
    ax_acc.plot(epochs, history.history["val_accuracy"], label="Validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()

    ax_loss.plot(epochs, history.history["loss"], label="Train")
    ax_loss.plot(epochs, history.history["val_loss"], label="Validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    fig.suptitle("Training History", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history saved to {output_path}")


def plot_confusion_matrix(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    output_path: str = "confusion_matrix.png",
) -> None:
    all_labels, all_preds = [], []
    for images, labels in val_dataset:
        preds = np.argmax(model.predict(images, verbose=0), axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_accuracy_vs_loss(
    history: tf.keras.callbacks.History,
    output_path: str = "accuracy_vs_loss.png",
) -> None:
    train_acc = history.history["accuracy"]
    train_loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc_train = ax.scatter(train_loss, train_acc, c=epochs, cmap="Blues", s=80, zorder=3, label="Train")
    ax.scatter(val_loss, val_acc, c=epochs, cmap="Oranges", s=80, zorder=3, marker="D", label="Validation")
    ax.plot(train_loss, train_acc, color="steelblue", alpha=0.4)
    ax.plot(val_loss, val_acc, color="darkorange", alpha=0.4)
    for i, e in enumerate(epochs):
        ax.annotate(str(e), (train_loss[i], train_acc[i]), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color="steelblue")
        ax.annotate(str(e), (val_loss[i], val_acc[i]), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color="darkorange")
    plt.colorbar(sc_train, ax=ax, label="Epoch (train)")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Loss per Epoch", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Accuracy vs loss chart saved to {output_path}")
