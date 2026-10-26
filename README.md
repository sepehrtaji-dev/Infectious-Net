# Face-infectious

A PyTorch image classifier that distinguishes **Chickenpox**, **Measles**, **Monkeypox** and **Normal** skin from photos, with a minimal desktop app for trying it on your own images.

> **Disclaimer:** This is an educational / research project. It is **not** a medical device and must not be used for diagnosis or treatment decisions. If you are worried about a skin condition, see a healthcare professional.

## Features

- Transfer learning with EfficientNet-B0 (default), ResNet-18 or ResNet-50
- Two-stage training: frozen backbone warm-up, then full fine-tuning with cosine LR schedule, label smoothing, class weighting, mixed precision and early stopping
- **Leakage-safe data splitting**: near-duplicate images are detected with a perceptual hash and kept in the same split
- Optional hold-out of a whole sub-dataset for a stricter test
- Minimal Tkinter app that shows the predicted class and per-class probabilities

## Project structure

```
Face-infectious/
├── app.py              # application entrypoint (launches the UI)
├── ui/
│   ├── __init__.py
│   └── main_window.py  # Tkinter window
├── dataset.py          # scanning, duplicate detection, splits, DataLoaders
├── model.py            # model builder + checkpoint helpers
├── train.py            # training + test evaluation
├── predict.py          # inference (Predictor class + CLI)
├── check_leakage.py    # verifies no near-duplicates between train/val and test
├── requirements.txt
├── checkpoints/        # best.pt and history.json are written here
└── Data/               # dataset (not included, see below)
```

## Dataset

Uses the [Infectious Skin Disease Dataset](https://www.kaggle.com/datasets/gupiii/infectious-skin-disease-dataset) from Kaggle. Download it and place it in `Data/` so it looks like:

```
Data/
├── Dataset_1/
│   ├── Chickenpox/  Measles/  Monkeypox/  Normal/
├── Dataset_2/
│   ├── Chickenpox/  Measles/  Monkeypox/  Normal/
└── Dataset_Gabungan/
    ├── Chickenpox/  Measles/  Monkeypox/  Normal/
```

Please check the dataset's own license and terms on Kaggle before redistributing it. The images are not part of this repository.

### Important: duplicates and data leakage

`Dataset_Gabungan` ("combined" in Indonesian) is a merge of `Dataset_1` and `Dataset_2`, and those two also overlap each other. A naive random split therefore puts identical images in both train and test and reports a misleadingly high accuracy (about 99.8% in our first run).

This project handles it by:

1. Excluding `Dataset_Gabungan` (`EXCLUDE_DIRS` in `dataset.py`)
2. Clustering near-duplicates with a 64-bit difference hash and splitting by cluster
3. Providing `check_leakage.py` to verify the result

After removing duplicates about 1,430 images remain (in about 750 duplicate-free groups).

## Installation

Python 3.10+ is required.

```bash
git clone https://github.com/sepehrtaji-dev/Face-infectious.git
cd Face-infectious
pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

```bash
# 1. Sanity-check the data pipeline (prints split sizes and class weights)
python dataset.py

# 2. Verify there is no leakage between train/val and test
python check_leakage.py

# 3. Train (saves checkpoints/best.pt)
python train.py
python train.py --backbone resnet50 --img-size 288 --epochs 30

# 4. Strict evaluation: test only on one sub-dataset, train on the rest
python train.py --holdout Dataset_2

# 5. Predict from the command line
python predict.py path/to/image.jpg

# 6. Launch the app
python app.py
```

Useful `train.py` options: `--backbone {efficientnet_b0,resnet18,resnet50}`, `--img-size`, `--batch-size`, `--epochs`, `--warmup-epochs`, `--lr-head`, `--lr-finetune`, `--patience`, `--workers` (use `0` if you hit DataLoader problems on Windows), `--holdout`.

## Results

EfficientNet-B0, default settings, duplicate-safe random split (217 test images):

| Class      | Recall |
|------------|--------|
| Chickenpox | 0.875  |
| Measles    | 0.846  |
| Monkeypox  | 0.819  |
| Normal     | 0.974  |
| **Overall accuracy** | **0.885** |

With only 217 test images the 95% confidence interval is roughly ±4 points, so small differences between runs are noise. Results on photos from other sources (for example phone cameras) will likely be lower than on this dataset.

## Limitations

- Small dataset after deduplication; the model overfits (training accuracy near 99%, validation around 91-92%)
- Monkeypox is sometimes confused with Chickenpox
- A few duplicate clusters contain the same image under different labels (label noise in the source data)
- Not clinically validated

## Roadmap

- Stronger augmentation and regularization
- Test-time augmentation
- Cross-validation / multiple seeds for more reliable estimates
- Hold-out evaluation on every sub-dataset

## License

Released under the [MIT License](LICENSE).