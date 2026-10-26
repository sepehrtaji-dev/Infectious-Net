import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASS_NAMES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def collect_samples(data_dir="Data"):
    lookup = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    samples = []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = lookup.get(p.parent.name.strip().lower())
            if label is not None:
                samples.append((p, label))
    if not samples:
        raise FileNotFoundError(f"No images found under '{data_dir}'. Check the folder layout.")
    return samples


def stratified_split(samples, val_frac=0.15, test_frac=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val, test = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    rng.shuffle(train)
    return train, val, test


def get_transforms(img_size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class SkinDiseaseDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(data_dir="Data", img_size=224, batch_size=32,
                    num_workers=4, seed=42):
    samples = collect_samples(data_dir)
    train_s, val_s, test_s = stratified_split(samples, seed=seed)

    train_ds = SkinDiseaseDataset(train_s, get_transforms(img_size, True))
    val_ds = SkinDiseaseDataset(val_s, get_transforms(img_size, False))
    test_ds = SkinDiseaseDataset(test_s, get_transforms(img_size, False))

    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(train_ds, batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin),
        "val": DataLoader(val_ds, batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin),
        "test": DataLoader(test_ds, batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin),
    }

    counts = Counter(label for _, label in train_s)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(CLASS_NAMES) * max(counts.get(i, 1), 1)) for i in range(len(CLASS_NAMES))],
        dtype=torch.float32,
    )
    info = {
        "train": len(train_s), "val": len(val_s), "test": len(test_s),
        "class_counts": {CLASS_NAMES[i]: counts.get(i, 0) for i in range(len(CLASS_NAMES))},
    }
    return loaders, weights, info


if __name__ == "__main__":
    _, w, info = get_dataloaders(num_workers=0)
    print(info)
    print("class weights:", w.tolist())