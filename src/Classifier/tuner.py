import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import itertools
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from model import ResNet18Model
from Data_Loader.data_loader import LIDCNoduleDataset
from trainer import train_one_epoch, validate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# CONFIG SEARCH SPACE
# =========================
SEARCH_SPACE = {
    "lr": [1e-3, 1e-4],
    "batch_size": [8, 16],
    "weight_decay": [0, 1e-4],
    "mode": ["2d", "2.5d"]   # 🔥 tambahan
}

EPOCHS = 5  # kecil dulu untuk tuning cepat
ROOT_DIR = "Dataset_Processed"

TRAIN_CSV = "Dataset_Processed/splits/train.csv"
VAL_CSV   = "Dataset_Processed/splits/val.csv"

SAVE_PATH = "best_params.txt"


# =========================
# RUN ONE EXPERIMENT
# =========================
def run_experiment(params):

    print(f"\nTesting config: {params}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    train_dataset = LIDCNoduleDataset(
        train_df,
        root_dir=ROOT_DIR,
        mode=params["mode"]
    )

    val_dataset = LIDCNoduleDataset(
        val_df,
        root_dir=ROOT_DIR,
        mode=params["mode"]
    )

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

    best_val_acc = 0

    # 🔥 tqdm epoch
    epoch_bar = tqdm(range(EPOCHS), desc="Epoch", leave=False)

    for epoch in epoch_bar:

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        val_metrics = validate(
            model, val_loader, criterion, DEVICE
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        # update tqdm display
        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "f1": f"{val_metrics['f1']:.4f}"
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


# =========================
# GRID SEARCH
# =========================
def grid_search():

    keys = SEARCH_SPACE.keys()
    values = SEARCH_SPACE.values()

    best_score = 0
    best_params = None

    for combination in itertools.product(*values):

        params = dict(zip(keys, combination))

        score = run_experiment(params)

        if score > best_score:
            best_score = score
            best_params = params

            print("\n🔥 New Best Found!")
            print(best_params, best_score)

    return best_params, best_score


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    best_params, best_score = grid_search()

    print("\n======================")
    print("BEST CONFIG:")
    print(best_params)
    print("BEST VAL ACC:", best_score)

    # save
    with open(SAVE_PATH, "w") as f:
        f.write(str(best_params))