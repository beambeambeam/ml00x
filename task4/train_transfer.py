"""GPU transfer-learning experiment for Task 4.

Uses ImageNet-pretrained EfficientNet-B0, mild screenshot-safe augmentation,
weighted sampling, and macro-F1 validation. Test images are never used for fit.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUBMIT = ROOT / "submit"
MODELS = ROOT / "models"
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch-cache"))
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["7-11k", "fiveN", "freefried", "pube", "roblock"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_image(image: Image.Image) -> Image.Image:
    return ImageOps.pad(image, (256, 256), method=Image.Resampling.LANCZOS, color=(0, 0, 0))


def make_transform(train: bool) -> transforms.Compose:
    ops: list[object] = [
        transforms.Lambda(pad_image),
    ]
    if train:
        ops += [
            transforms.Resize((128, 128)),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        ]
    else:
        ops += [transforms.Resize((128, 128))]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class GameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, folder: Path, transform: transforms.Compose, label_map: dict[str, int] | None = None):
        self.frame = frame.reset_index(drop=True)
        self.folder = folder
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.folder / f"{row.id}.jpg") as image:
            x = self.transform(image.convert("RGB"))
        if self.label_map is None:
            return x, row.id
        return x, self.label_map[row["class"]]


def make_model() -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASSES))
    return model.to(DEVICE)


def loader(dataset: Dataset, train: bool, batch_size: int = 64) -> DataLoader:
    kwargs = dict(batch_size=128, num_workers=4, pin_memory=DEVICE.type == "cuda", persistent_workers=True, prefetch_factor=2)
    if train:
        labels = np.array([dataset.label_map[c] for c in dataset.frame["class"]])
        counts = np.bincount(labels, minlength=len(CLASSES))
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)
        return DataLoader(dataset, sampler=sampler, **kwargs)
    return DataLoader(dataset, shuffle=False, **kwargs)


def run_epoch(model, dl, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    all_y, all_p, total_loss = [], [], 0.0
    for x, y in dl:
        x = x.to(DEVICE, non_blocking=True)
        y = torch.as_tensor(y, device=DEVICE)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        all_y.extend(y.detach().cpu().numpy())
        all_p.extend(logits.argmax(1).detach().cpu().numpy())
    return total_loss / len(dl.dataset), f1_score(all_y, all_p, average="macro")


@torch.no_grad()
def predict(model, dl):
    model.eval()
    out = []
    for x, _ in dl:
        out.extend(model(x.to(DEVICE, non_blocking=True)).softmax(1).cpu().numpy())
    return np.asarray(out)


def train_model(frame: pd.DataFrame, epochs: int, validation: bool):
    label_map = {c: i for i, c in enumerate(CLASSES)}
    if validation:
        tr_idx, va_idx = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=frame["class"], random_state=RANDOM_STATE)
        tr_frame, va_frame = frame.iloc[tr_idx], frame.iloc[va_idx]
    else:
        tr_frame, va_frame = frame, None
    train_ds = GameDataset(tr_frame, DATA / "train", make_transform(True), label_map)
    train_dl = loader(train_ds, True)
    val_dl = None
    if va_frame is not None:
        val_dl = loader(GameDataset(va_frame, DATA / "train", make_transform(False), label_map), False)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        loss, train_f1 = run_epoch(model, train_dl, optimizer)
        val_f1 = train_f1 if val_dl is None else run_epoch(model, val_dl)[1]
        print(f"epoch {epoch}/{epochs}: loss={loss:.4f} train_f1={train_f1:.4f} val_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    seed_everything()
    print(f"device={DEVICE}")
    train = pd.read_csv(DATA / "train.csv")
    train["class"] = train["class"].str.strip()
    test = pd.read_csv(DATA / "test.csv")
    model, val_f1 = train_model(train, epochs=5, validation=True)
    print(f"best validation macro-F1: {val_f1:.5f}")
    if val_f1 <= 0.46307:
        print("validation did not beat the existing 0.46307 baseline; no submission replaced")
        return
    # Retrain from pretrained weights on all labelled rows using the selected setup.
    model, _ = train_model(train, epochs=5, validation=False)
    test_ds = GameDataset(test, DATA / "test", make_transform(False), None)
    probs = predict(model, loader(test_ds, False))
    pred = np.array([CLASSES[i] for i in probs.argmax(1)])
    pred[pred == "7-11k"] = "7-11K"
    sample = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    assert len(pred) == len(sample) == 25889
    SUBMIT.mkdir(exist_ok=True)
    pd.DataFrame({"id": sample["id"], "task4": pred}).to_csv(SUBMIT / "index.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / "resnet18-task4.pt")
    print(pd.Series(pred).value_counts().to_string())
    print(f"wrote {SUBMIT / 'index.csv'}")


if __name__ == "__main__":
    main()
