import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASS_NAMES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDE_DIRS = {"dataset_gabungan"}      
DUP_THRESHOLD = 4                         
HASH_CACHE = ".hash_cache.json"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def collect_samples(data_dir="Data", exclude=EXCLUDE_DIRS):
    lookup = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    exclude = {e.lower() for e in (exclude or ())}
    samples = []
    for p in sorted(Path(data_dir).rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        if any(part.lower() in exclude for part in p.parts):
            continue
        label = lookup.get(p.parent.name.strip().lower())
        if label is not None:
            samples.append((p, label))
    if not samples:
        raise FileNotFoundError(f"No images found under '{data_dir}'. Check the folder layout.")
    return samples


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = img.tobytes()
    bits = 0
    for r in range(size):
        for c in range(size):
            i = r * (size + 1) + c
            bits = (bits << 1) | int(px[i] > px[i + 1])
    return bits


def hash_all(samples, cache_file=HASH_CACHE):
    cf = Path(cache_file)
    cache = {}
    if cf.exists():
        try:
            cache = json.loads(cf.read_text())
        except Exception:
            cache = {}
    changed, out = False, {}
    for p, _ in samples:
        k = str(p)
        if k not in cache:
            cache[k] = dhash(p)
            changed = True
        out[p] = cache[k]
    if changed:
        cf.write_text(json.dumps(cache))
    return out


def near_duplicate_groups(samples, threshold=DUP_THRESHOLD):
    hashes = hash_all(samples)
    hs = [hashes[p] for p, _ in samples]
    n = len(hs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        hi = hs[i]
        for j in range(i + 1, n):
            if (hi ^ hs[j]).bit_count() <= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return [find(i) for i in range(n)]


def group_split(samples, groups, val_frac=0.15, test_frac=0.15, seed=42):
    rng = random.Random(seed)
    clusters = {}
    for s, g in zip(samples, groups):
        clusters.setdefault(g, []).append(s)

    by_class, mixed = {}, 0
    for members in clusters.values():
        labels = Counter(l for _, l in members)
        if len(labels) > 1:
            mixed += 1
        by_class.setdefault(labels.most_common(1)[0][0], []).append(members)
    if mixed:
        print(f"WARNING: {mixed} duplicate clusters contain images from DIFFERENT classes "
              f"(label noise in the dataset).")

    train, val, test = [], [], []
    for cl in by_class.values():
        rng.shuffle(cl)
        total = sum(len(m) for m in cl)
        want_test, want_val = test_frac * total, val_frac * total
        t = v = 0
        for m in cl:
            if t < want_test:
                test += m
                t += len(m)
            elif v < want_val:
                val += m
                v += len(m)
            else:
                train += m
    rng.shuffle(train)
    return train, val, test


def make_splits(data_dir="Data", holdout=None, seed=42):
    samples = collect_samples(data_dir)

    if not holdout:
        groups = near_duplicate_groups(samples)
        print(f"{len(samples)} images in {len(set(groups))} duplicate-free groups")
        return group_split(samples, groups, seed=seed)

    key = holdout.lower()
    test_all = [s for s in samples if key in (part.lower() for part in s[0].parts)]
    if not test_all:
        raise ValueError(f"Holdout folder '{holdout}' not found (or it is excluded).")
    test_set = set(test_all)
    rest = [s for s in samples if s not in test_set]

    groups = near_duplicate_groups(rest)
    train_s, val_s, _ = group_split(rest, groups, val_frac=0.15, test_frac=0.0, seed=seed)

    hashes = hash_all(samples)
    rest_h = [hashes[p] for p, _ in rest]
    test_s = [s for s in test_all
              if not any((hashes[s[0]] ^ h).bit_count() <= DUP_THRESHOLD for h in rest_h)]
    print(f"holdout '{holdout}': kept {len(test_s)}/{len(test_all)} test images "
          f"(dropped {len(test_all) - len(test_s)} that also exist in the training data)")
    return train_s, val_s, test_s


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
                    num_workers=4, seed=42, holdout=None):
    train_s, val_s, test_s = make_splits(data_dir, holdout, seed)

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