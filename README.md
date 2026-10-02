# Road Crack Detection with a CNN (Image-Level Classification)

A lightweight convolutional neural network that classifies road-surface images as **crack** or **no crack**. Built as a final-year Bachelor's thesis project at Ekiti State University (2025), and designed to be trained on the free Google Colab tier and run on a CPU.

## What this is (and isn't)

- **Task:** binary image classification. The model outputs one probability per image: how likely it is that the image contains a crack.
- **Not included:** crack localisation or pixel-level segmentation. The model does not say *where* the crack is.

## Model

Four Conv2D blocks (32, 64, 128, 256 filters, each followed by BatchNorm and 2x2 max-pooling), then global average pooling, two dense layers (512, 256) with dropout, and a sigmoid output. Input is 224x224 RGB, scaled to [0, 1]. Trained with binary cross-entropy and Adam, with class weighting, early stopping, and learning-rate reduction on validation loss.

## Repository contents

| File | Purpose |
|---|---|
| `train.py` | Trains the model on `data/train` and `data/val`; saves the best checkpoint and class order |
| `evaluate.py` | Evaluates a trained model on a held-out `data/test` folder (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, optional speed benchmark) |
| `inference.py` | Predicts crack / no crack for one image or a folder of images |
| `requirements.txt` | Python dependencies |

## Data layout

```
data/
├── train/<class folders>/
├── val/<class folders>/
└── test/<class folders>/     # held-out; used only by evaluate.py
```

Split by road segment or collection session, not by random image, so near-duplicate frames do not appear on both sides of a split.

> **Label order:** Keras assigns class indices alphabetically by folder name, and the model output is the probability of class index 1. `train.py` prints the order and saves it to `class_names.json`. Use the `--positive-class` flag in `evaluate.py` and `--crack-is-class-1` in `inference.py` to match your folder names.

## Installation

```bash
git clone https://github.com/SamsondareHUB/Crack-dectection-CNN.git
cd Crack-dectection-CNN
pip install -r requirements.txt
```

## Usage

```bash
# Train (Colab with a GPU recommended)
python train.py --data data

# Evaluate on the held-out test set
python evaluate.py --model best_crack_model.keras --test-dir data/test --positive-class crack --benchmark

# Predict on a single image or a folder
python inference.py --model best_crack_model.keras --image path/to/road.jpg
python inference.py --model best_crack_model.keras --folder path/to/images --output results.csv
```

## Results

Held-out test-set results will be added here once the re-evaluation with the scripts above is complete. Results from earlier experiments are not reported because they were not measured on a separate test split.

## Limitations

- Trained on road images from a limited set of locations; performance on other road types, lighting conditions, camera angles, or regions is untested.
- Image-level output only: no localisation, crack width, or severity.

## License

MIT. See `LICENSE`.

## Author

Samson Oluwadare, Department of Computer Science, Ekiti State University.
