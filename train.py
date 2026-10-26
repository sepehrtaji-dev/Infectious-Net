"""Train the classifier.  Example:  python train.py --backbone efficientnet_b0 --epochs 25"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from Dataset import CLASS_NAMES, get_dataloaders
from FaceNet import BACKBONES, build_model, save_checkpoint, set_backbone_trainable


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    k = len(CLASS_NAMES)
    cm = torch.zeros(k, k, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y, pred):
            cm[t, p] += 1
    acc = cm.diagonal().sum().item() / cm.sum().item()
    print(f"\nTest accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + " ".join(f"{n[:8]:>9}" for n in CLASS_NAMES))
    for i, row in enumerate(cm.tolist()):
        recall = row[i] / max(sum(row), 1)
        print(f"{CLASS_NAMES[i][:10]:>11} " + " ".join(f"{v:>9}" for v in row) + f"   recall={recall:.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--backbone", default="efficientnet_b0", choices=BACKBONES)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup-epochs", type=int, default=3, help="epochs with frozen backbone")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.manual_seed(42)

    loaders, class_weights, info = get_dataloaders(
        args.data, args.img_size, args.batch_size, args.workers)
    print("Data:", info)

    model = build_model(len(CLASS_NAMES), args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def make_opt(lr):
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        return opt, None

    set_backbone_trainable(model, False)
    optimizer, scheduler = make_opt(args.lr_head)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    best_path = out_dir / "best.pt"
    best_acc, bad_epochs, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:  # switch to fine-tuning
            set_backbone_trainable(model, True)
            optimizer, _ = make_opt(args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
            print("-> backbone unfrozen, fine-tuning")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        if scheduler:
            scheduler.step()

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        print(f"Epoch {epoch:02d}/{args.epochs} | train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f} | {time.time() - t0:.0f}s")

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            save_checkpoint(best_path, model, args.backbone, CLASS_NAMES, args.img_size,
                            extra={"val_acc": va_acc})
            print(f"   saved best (val_acc={va_acc:.4f})")
        else:
            bad_epochs += 1
            if epoch > args.warmup_epochs and bad_epochs >= args.patience:
                print("Early stopping.")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Evaluate the best checkpoint on the held-out test split
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    evaluate_test(model, loaders["test"], device)


if __name__ == "__main__":
    main()