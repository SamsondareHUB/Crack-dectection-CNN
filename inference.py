
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-dir", required=True, help="folder with one subfolder per class")
    ap.add_argument("--positive-class", required=True, help="folder name of the crack class")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--benchmark", action="store_true", help="measure CPU images/s, batch size 1")
    args = ap.parse_args()

    ds = tf.keras.utils.image_dataset_from_directory(
        args.test_dir, labels="inferred", label_mode="int",
        image_size=(args.img_size, args.img_size), batch_size=32, shuffle=False)
    names = ds.class_names
    print("Class order (index -> name):", dict(enumerate(names)))
    if args.positive_class not in names:
        raise SystemExit(f"--positive-class must be one of {names}")
    if len(names) != 2:
        raise SystemExit("Expected exactly two class folders.")

    model = tf.keras.models.load_model(args.model)
    ds_scaled = ds.map(lambda x, y: (x / 255.0, y))
    y_idx = np.concatenate([y.numpy() for _, y in ds])
    p_class1 = model.predict(ds_scaled, verbose=0).reshape(-1)  # model output = P(class index 1)

    pos_idx = names.index(args.positive_class)
    p_crack = p_class1 if pos_idx == 1 else 1.0 - p_class1
    y_true = (y_idx == pos_idx).astype(int)
    y_pred = (p_crack >= args.threshold).astype(int)

    print(f"\nN = {len(y_true)}  (crack: {y_true.sum()}, no crack: {(1 - y_true).sum()})")
    print(classification_report(y_true, y_pred, target_names=["No Crack", "Crack"], digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_true, p_crack):.4f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_true, p_crack):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Crack", "Crack"], yticklabels=["No Crack", "Crack"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix (test set)")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

    if args.benchmark:
        x = tf.random.uniform((1, args.img_size, args.img_size, 3))
        for _ in range(10):  # warm-up
            model(x, training=False)
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, training=False)
        dt = time.perf_counter() - t0
        print(f"\nModel-only throughput: {n / dt:.1f} images/s (batch 1, "
              f"excludes image loading/preprocessing; run on the target CPU)")


if __name__ == "__main__":
    main()
