
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class CrackDetectionInference:
    def __init__(self, model_path, img_height=224, img_width=224, crack_is_class_1=True):
        """
        crack_is_class_1: the model's sigmoid output is P(class index 1).
        Keras orders classes alphabetically by folder name, so with folders
        'crack' / 'no_crack' the output is P(no_crack) -> pass False.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.crack_is_class_1 = crack_is_class_1
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading model from {model_path}...")
        return tf.keras.models.load_model(model_path)

    def preprocess_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        return img.astype("float32") / 255.0

    def crack_probability(self, batch):
        p = self.model.predict(batch, verbose=0).reshape(-1)
        return p if self.crack_is_class_1 else 1.0 - p

    def predict_single_image(self, image_path, threshold=0.5):
        p_crack = float(self.crack_probability(self.preprocess_image(image_path)[None])[0])
        label = "CRACK" if p_crack >= threshold else "NO CRACK"
        confidence = p_crack if label == "CRACK" else 1.0 - p_crack
        return label, confidence, p_crack

    def batch_predict(self, folder, output_file=None, threshold=0.5, batch_size=32):
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        # set + lowercase suffix check avoids double-counting on case-insensitive filesystems
        files = sorted({p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS})
        if not files:
            print(f"No image files found in {folder}")
            return []

        results = []
        for start in range(0, len(files), batch_size):
            chunk, valid = [], []
            for f in files[start:start + batch_size]:
                try:
                    chunk.append(self.preprocess_image(f))
                    valid.append(f)
                except ValueError as e:
                    print(f"Skipping: {e}")
            if not chunk:
                continue
            probs = self.crack_probability(np.stack(chunk))
            for f, p in zip(valid, probs):
                label = "CRACK" if p >= threshold else "NO CRACK"
                results.append({"image": f.name, "prediction": label, "p_crack": float(p)})
                print(f"{f.name}: {label} (P(crack)={p:.3f})")

        if output_file:
            with open(output_file, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["image", "prediction", "p_crack"])
                w.writeheader()
                w.writerows(results)
            print(f"Results saved to {output_file}")
        return results


def main():
    ap = argparse.ArgumentParser(description="Crack detection inference")
    ap.add_argument("--model", default="best_crack_detection_model.h5")
    ap.add_argument("--image")
    ap.add_argument("--folder")
    ap.add_argument("--output")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--crack-is-class-1", type=lambda s: s.lower() in {"1", "true", "yes"},
                    default=True, help="True if model output = P(crack). With folders "
                    "'crack'/'no_crack' this should be False.")
    args = ap.parse_args()

    detector = CrackDetectionInference(args.model, crack_is_class_1=args.crack_is_class_1)
    if args.image:
        label, conf, p = detector.predict_single_image(args.image, args.threshold)
        print(f"{args.image}: {label} (confidence {conf:.2%}, P(crack)={p:.3f})")
    elif args.folder:
        results = detector.batch_predict(args.folder, args.output, args.threshold)
        n = sum(r["prediction"] == "CRACK" for r in results)
        print(f"\nSummary: {n}/{len(results)} images predicted as cracked")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
