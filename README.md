# Lightweight Road Crack Segmentation with a Depthwise-Separable U-Net

Pixel-level road crack segmentation designed for low-resource settings: trained on the free Google Colab tier and run on a CPU. Developed as a final-year Bachelor's thesis at Ekiti State University (2025).

<!-- Add a hero image here: input | ground-truth mask | prediction overlay -->

## Overview

- **Task:** binary semantic segmentation (crack vs. background)
- **Model:** U-Net variant with depthwise separable convolutions, reducing parameters by ~65% relative to a standard U-Net (`<N>` M vs `<M>` M parameters)
- **Data:** 2,000+ images of Nigerian roads (Ekiti-Ondo highways) with pixel-level annotations, captured under dust, shadow, and poor-lighting conditions
- **Training:** TensorFlow 2.x / Keras on Colab (T4 free tier)
- **Inference:** ~28 FPS on an Intel i5 (8th gen) CPU, `<input size>` input, batch size 1

## Results

| Model | Params | mIoU | Crack-class IoU | F1 | CPU FPS |
|---|---|---|---|---|---|
| Standard U-Net (baseline) | `<>` | `<>` | `<>` | `<>` | `<>` |
| **Ours (DS-U-Net)** | `<>` | `<96.4>` | `<>` | `<>` | `<28>` |

Evaluation protocol: `<describe split: counts, how split was made (by image / by road segment / by location), threshold used, how mIoU is computed>`.

> Note: mIoU averages crack and background classes, so it is inflated when cracks cover few pixels. Report crack-class IoU/F1 alongside it.

## Repository structure

```
.
├── train.py            # training script  (rename from "Crack-dectection system CNN.py")
├── inference.py        # run the model on an image or folder
├── requirements.txt
├── samples/            # a few example images + predictions
└── LICENSE
```

## Installation

```bash
git clone https://github.com/SamsondareHUB/Crack-dectection-CNN.git
cd Crack-dectection-CNN
pip install -r requirements.txt
```

## Usage

```bash
# Inference on one image
python inference.py --image samples/test_road.jpg --weights <path/to/weights>

# Training (Colab recommended)
python train.py --data <path/to/data> --epochs <N>
```

## Dataset

`<Available / not publicly available>`. Images were collected `<how, with what device, when>` and annotated `<tool, annotator count, QA process>`.

Expected layout:

```
data/
├── raw/      # RGB images
└── masks/    # binary masks (crack = 255)
```

Augmentations (Albumentations): rotation, flip, brightness, contrast, blur.

## Limitations

- Trained on Nigerian highway imagery from a limited number of locations; generalisation to other surfaces or regions is untested.
- `<other known failure cases, e.g. shadows, patched asphalt, wet roads>`

## License

MIT. See `LICENSE`.

## Citation / Contact

Samson Oluwadare, Department of Computer Science, Ekiti State University.
