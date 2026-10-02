
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(img_size=224):
    return keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"), layers.Dropout(0.5),
        layers.Dense(256, activation="relu"), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])


def load_split(path, img_size, batch_size, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="binary",
        image_size=(img_size, img_size), batch_size=batch_size,
        shuffle=shuffle, seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder containing train/ and val/")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="best_crack_model.keras")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(42)

    train_raw = load_split(f"{args.data}/train", args.img_size, args.batch_size, True)
    val_raw = load_split(f"{args.data}/val", args.img_size, args.batch_size, False)
    class_names = train_raw.class_names
    assert class_names == val_raw.class_names, "train/val class folders differ"
    print("Class order (index -> name):", dict(enumerate(class_names)))
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # Class weights from the training labels (handles imbalance)
    y = np.concatenate([lbl.numpy().ravel() for _, lbl in train_raw]).astype(int)
    counts = np.bincount(y, minlength=2)
    class_weight = {i: len(y) / (2.0 * counts[i]) for i in range(2)}
    print("Train counts:", dict(zip(class_names, counts.tolist())), "| class weights:", class_weight)

    # Augmentation only on training data; scaling to [0,1] done in the pipeline so the
    # saved model expects [0,1] inputs, matching inference.py / evaluate.py.
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(20 / 360),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])
    autotune = tf.data.AUTOTUNE
    train_ds = (train_raw.map(lambda x, l: (augment(x / 255.0, training=True), l), num_parallel_calls=autotune)
                .prefetch(autotune))
    val_ds = val_raw.map(lambda x, l: (x / 255.0, l), num_parallel_calls=autotune).prefetch(autotune)

    model = build_model(args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")])
    model.summary()

    callbacks = [
        # Same metric everywhere, so the checkpoint is the model early stopping restores.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weight, callbacks=callbacks)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, key in zip(axes.ravel(), ["accuracy", "loss", "precision", "recall"]):
        ax.plot(history.history[key], label=f"train {key}")
        ax.plot(history.history[f"val_{key}"], label=f"val {key}")
        ax.set_title(key.capitalize()); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
    print(f"Best model saved to {args.out}. Evaluate it on data/test with evaluate.py.")


if __name__ == "__main__":
    main()
