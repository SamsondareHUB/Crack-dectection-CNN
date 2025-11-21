# Road Crack Detection Using Deep CNNs 

Real-time, lightweight road crack detection system built for resource-constrained environments (especially developing countries). Achieved *96.4% mean IoU* on a custom Nigerian road dataset using a modified U-Net architecture — outperforming many published results under similar hardware limits.

# Project Highlights
- Custom dataset of *2,000+ annotated Nigerian road images* collected under real-world conditions (dust, shadows, poor lighting)
- Modified U-Net with depthwise separable convolutions → *65% fewer parameters* than standard U-Net
- Trained entirely on *Google Colab Free tier* (no local GPU!)
- Achieved *96.4% mIoU, **97.8% accuracy* on test set
- Runs inference at *28 FPS* on CPU (Intel i5-8th Gen)
- Final-year Bachelor's thesis project – Ekiti State University, 2025

# Tech Stack
- Python, TensorFlow 2.x / Keras
- OpenCV, Albumentations
- Google Colab (training), Jupyter Notebook
- Matplotlib, Seaborn (visualization)

# Dataset
- data/raw/ – Original images from Ekiti-Ondo highways
- data/masks/ – Pixel-level binary annotations (crack = white)
- Augmentations: rotation, flip, brightness, contrast, blur

# Quick Start
```bash
git clone https://github.com/SamsondareHUB/crack-detection-CNN
cd crack-detection
pip install -r requirements.txt

# Run inference on a single image
python predict.py --image samples/test_road.jpg

# Train the model (Colab recommended)
jupyter notebook train.ipynb
