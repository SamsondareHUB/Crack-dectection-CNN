# Road Crack Detection with a CNN (Image-Level Classification)

A lightweight convolutional neural network that classifies road-surface images as **crack** or **no crack**. Built as a final-year Bachelor's thesis project at Ekiti State University (2025), and designed to be trained on the free Google Colab tier and run on a CPU.

<!-- Add a figure here: a grid of correctly/incorrectly classified example images -->

## What this is (and isn't)

- **Task:** binary image classification. The model outputs one probability per image: how likely it is that the image contains a crack.
- **Not included:** crack localisation or pixel-level segmentation. The model does not say *where* the crack is.

## Model

Four Conv2D blocks (32 → 64 → 128 → 256 filters, each with BatchNorm and 2×2 max-pooling), followed by global average pooling, two dense layers (512, 256) with dropout, and a sigmoid output. Input size is 224×224 RGB, pixel values scaled to [0, 1]. Trained with binary cross-entropy and Adam, with early stopping and learning-rate reduction on validation loss.

Parameters: `<N>` (run `model.summary()`).

## Data

`<Describe: number of images per class, where and how they were collected (device, locations, conditions), how labels were assigned, and whether the dataset can be shared.>`

Expected folder layout (class names are examples; see the note on label order below):

```
data/
├── train/
│   ├── crack/
│   └── no_crack/
└── test/            # held-out, never used for tuning or early stopping
    ├── crack/
    └── no_crack/
```

> **Label order:** Keras assigns class indices alphabetically by folder name, and the model's single output is the probability of class index 1. With `crack` / `no_crack`, index 1 is **no_crack**. `evaluate.py` and `inference.py` handle this explicitly, so set the flags correctly.

## Results

Evaluated on a held-out test set of `<N>` images (`<N_crack>` crack / `<N_no_crack>` no crack), split by `<road segment / location / collection session>` so that near-duplicate frames do not appear in both train and test.

| Metric (crack = positive class) | Value |
|---|---|
| Accuracy | `<>` |
| Precision | `<>` |
| Recall | `<>` |
| F1 | `<>` |
| ROC-AUC | `<>` |
| PR-AUC | `<>` |

Generate all of these with `evaluate.py` (below). Inference speed: `<X>` images/s on `<CPU model>`, batch size 1, measured with `evaluate.py --benchmark`.

## Installation

```bash
git clone https://github.com/SamsondareHUB/Crack-dectection-CNN.git
cd Crack-dectection-CNN
pip install -r requirements.txt
```

## Usage

```bash
# Single image
python inference.py --model best_crack_detection_model.h5 --image path/to/road.jpg

# Folder of images, saving a CSV
python inference.py --model best_crack_detection_model.h5 --folder path/to/images --output results.csv

# Full evaluation on the held-out test set
python evaluate.py --model best_crack_detection_model.h5 --test-dir data/test --positive-class crack --benchmark
```

## Limitations

- Trained on `<region/roads>`; performance on other road types, lighting, camera angles, or regions is untested.
- Image-level output only: no localisation, no crack severity or width.
- `<Known failure cases, e.g. shadows, patched asphalt, road markings, wet surfaces. Add examples from your error analysis.>`

## License

MIT. See `LICENSE`.

## Author

Samson Oluwadare, Department of Computer Science, Ekiti State University.
