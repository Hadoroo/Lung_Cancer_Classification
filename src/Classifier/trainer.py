import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ast

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from model import ResNet18Model
from Data_Loader.data_loader import LIDCNoduleDataset

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "Dataset_Processed"
SPLIT_DIR = "Dataset_Processed/splits"
KFOLD_DIR = os.path.join(SPLIT_DIR, "kfold")

BEST_PARAM_PATH = "best_params_2.5d.txt"

EPOCHS = 10
MODE = "2d"   # 🔥 ubah ke "2d" untuk baseline


# =========================
# LOAD PARAMS
# =========================
def load_best_params():
    with open(BEST_PARAM_PATH, "r") as f:
        return ast.literal_eval(f.read())


# =========================
# TRAIN
# =========================
def train_one_epoch(model, loader, optimizer, criterion, DEVICE):
    model.train()
    total_loss = 0

    loop = tqdm(loader, leave=False, desc="Train")

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # update tqdm
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# =========================
# VALIDATE (FULL METRICS)
# =========================
def validate(model, loader, criterion, DEVICE):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    loop = tqdm(loader, leave=False, desc="Val")

    with torch.no_grad():
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            loop.set_postfix(loss=f"{loss.item():.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

# =========================
# TRAIN 1 FOLD
# =========================
def train_fold(train_csv, val_csv, params, fold_idx):

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_dataset = LIDCNoduleDataset(train_df, ROOT_DIR, mode=MODE)
    val_dataset   = LIDCNoduleDataset(val_df, ROOT_DIR, mode=MODE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=4
    )

    model = ResNet18Model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    best_metrics = None
    best_acc = 0

    epoch_bar = tqdm(range(EPOCHS), desc=f"Fold {fold_idx}", leave=True)

    for epoch in epoch_bar:

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE)

        epoch_bar.set_postfix({
            "loss": f"{val_metrics['loss']:.4f}",
            "acc": f"{val_metrics['acc']:.4f}",
            "f1": f"{val_metrics['f1']:.4f}"
        })

        print(
            f"[Fold {fold_idx}] Epoch {epoch+1} | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['acc']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_metrics = val_metrics

            torch.save(
                model.state_dict(),
                f"best_model_fold_{fold_idx}_{MODE}.pth"
            )
    return best_metrics


# =========================
# K-FOLD TRAIN
# =========================
def run_kfold():

    params = load_best_params()

    print("Using params:", params)
    print("Mode:", MODE)

    results = []

    for i in range(1, 6):

        train_csv = os.path.join(KFOLD_DIR, f"fold_{i}_train.csv")
        val_csv   = os.path.join(KFOLD_DIR, f"fold_{i}_val.csv")

        metrics = train_fold(train_csv, val_csv, params, i)

        metrics["fold"] = i
        results.append(metrics)

    df = pd.DataFrame(results)
    save_path = f"kfold_results_{MODE}.csv"
    df.to_csv(save_path, index=False)

    print("\n=== K-FOLD RESULT ===")
    print(df)
    print("\nMEAN:")
    print(df.mean())


# =========================
# FINAL TEST
# =========================
def run_test():

    test_df = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    dataset = LIDCNoduleDataset(test_df, ROOT_DIR, mode=MODE)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = ResNet18Model().to(DEVICE)
    model.load_state_dict(torch.load(f"best_model_fold_1_{MODE}.pth"))

    criterion = nn.CrossEntropyLoss()

    metrics = validate(model, loader, criterion, DEVICE)

    print("\n=== FINAL TEST ===")
    print(metrics)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("=== TRAINING ===")
    run_kfold()

    print("\n=== TEST ===")
    run_test()