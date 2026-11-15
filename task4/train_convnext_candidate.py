"""Separate ConvNeXt-V2 candidate experiment for Task 4.

This never overwrites submit/index.csv. It writes a candidate submission under
predictions/ so the current ResNet18 result remains easy to roll back to.
"""

from __future__ import annotations

import copy
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageEnhance, ImageFile
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PREDICTIONS = ROOT / "predictions"
MODELS = ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS / "huggingface-cache"))
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["7-11K", "fiveN", "freefried", "pube", "roblock"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
CLASS_TO_INDEX["7-11k"] = 0
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def change_brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def make_transform(train: bool, brightness: float | None = None):
    ops: list[object] = [transforms.Resize((128, 256))]
    if train:
        ops += [
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.08),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if brightness is not None:
        ops.insert(1, transforms.Lambda(partial(change_brightness, factor=brightness)))
    return transforms.Compose(ops)


class GameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, folder: Path, transform, labelled: bool):
        self.frame = frame.reset_index(drop=True)
        self.folder = folder
        self.transform = transform
        self.labelled = labelled

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.folder / f"{row.id}.jpg") as image:
            x = self.transform(image.convert("RGB"))
        if self.labelled:
            return x, CLASS_TO_INDEX[row["class"]]
        return x, row.id


def build_model():
    model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=5)
    return model.to(DEVICE)


def make_loader(dataset, train: bool):
    return DataLoader(dataset, batch_size=32, shuffle=train, num_workers=4, pin_memory=True, persistent_workers=True)


def predict(model, loader):
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(model(x.to(DEVICE, non_blocking=True)).softmax(1).cpu().numpy())
    return np.vstack(chunks)


def calibrate(y_true, probabilities):
    weights = np.ones(5, dtype=np.float32)
    best = f1_score(y_true, (probabilities * weights).argmax(1), average="macro")
    for _ in range(3):
        for class_index in range(5):
            for value in np.linspace(0.80, 1.25, 46):
                trial = weights.copy()
                trial[class_index] = value
                score = f1_score(y_true, (probabilities * trial).argmax(1), average="macro")
                if score > best:
                    best, weights = score, trial
    return weights, best


def main() -> None:
    seed_everything()
    print(f"device={DEVICE}")
    train = pd.read_csv(DATA / "train.csv")
    train["class"] = train["class"].str.strip().replace({"7-11k": "7-11K"})
    test = pd.read_csv(DATA / "test.csv")
    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.2, stratify=train["class"], random_state=SEED)
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    train_loader = make_loader(GameDataset(tr, DATA / "train", make_transform(True), True), True)
    val_loader = make_loader(GameDataset(va, DATA / "train", make_transform(False), True), False)
    model = build_model()
    counts = np.bincount([CLASS_TO_INDEX[x] for x in tr["class"]], minlength=5)
    class_weights = torch.tensor(len(tr) / (5.0 * counts), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4 * len(train_loader), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    best_f1, best_state = -1.0, None
    y_val = np.array([CLASS_TO_INDEX[x] for x in va["class"]])
    for epoch in range(1, 5):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = torch.as_tensor(y, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        val_probs = predict(model, val_loader)
        val_score = f1_score(y_val, val_probs.argmax(1), average="macro")
        print(f"epoch {epoch}/4: val_macro_f1={val_score:.5f}", flush=True)
        if val_score > best_f1:
            best_f1, best_state = val_score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val_probs = predict(model, val_loader)
    calibration, calibrated_f1 = calibrate(y_val, val_probs)
    print(f"raw validation macro-F1: {best_f1:.5f}")
    print(f"calibrated validation macro-F1: {calibrated_f1:.5f}; weights={calibration.round(3)}")

    # TTA uses brightness shifts only; no horizontal flip because screenshots contain UI text.
    test_probs = []
    for brightness in (None, 0.90, 1.10):
        ds = GameDataset(test, DATA / "test", make_transform(False, brightness), False)
        test_probs.append(predict(model, make_loader(ds, False)))
    final_pred = (np.mean(test_probs, axis=0) * calibration).argmax(1)
    labels = np.array(CLASSES)[final_pred]
    PREDICTIONS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / "convnextv2-tiny-candidate.pt")
    sample = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    candidate = pd.DataFrame({"id": sample["id"], "task4": labels})
    candidate.to_csv(PREDICTIONS / "convnextv2-candidate-index.csv", index=False)
    print(candidate["task4"].value_counts().to_string())
    print(f"wrote {PREDICTIONS / 'convnextv2-candidate-index.csv'}")


if __name__ == "__main__":
    main()
