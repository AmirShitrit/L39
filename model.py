import tensorflow as tf

from config import EPOCHS, IMG_SIZE


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
