import sys
from pathlib import Path

import torch
from PIL import Image

from Dataset import get_transforms
from FaceNet import load_checkpoint

DEFAULT_CKPT = Path("checkpoints") / "best.pt"


class Predictor:
    def __init__(self, ckpt_path=DEFAULT_CKPT):
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found. Run train.py first.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.class_names = ckpt["class_names"]
        self.transform = get_transforms(ckpt["img_size"], train=False)

    @torch.no_grad()
    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().tolist()
        return sorted(zip(self.class_names, probs), key=lambda t: t[1], reverse=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <image_path>")
    for name, p in Predictor().predict(sys.argv[1]):
        print(f"{name:<12} {p * 100:6.2f}%")