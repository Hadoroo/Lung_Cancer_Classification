import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

import pandas as pd
import numpy as np

import sys
import os
import ast

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import (
    ResNet2D,
    ResNet3D
)

from Data_Loader.data_loader import (
    Dataset2D,
    Dataset3D
)

# =========================
# CONFIG
# =========================
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.backends.cudnn.benchmark = True

ROOT_DIR = "Dataset_Processed"

SPLIT_DIR = os.path.join(
    ROOT_DIR,
    "splits"
)

KFOLD_DIR = os.path.join(
    SPLIT_DIR,
    "kfold"
)

EPOCHS = 10

DEPTH = 16

# =========================
# MODE LIST
# =========================
MODES = [

    "2d",

    "2.5d",

    "3d"
]

# =========================
# SAVE DIRECTORY
# =========================
MODEL_DIR = "saved_models"

RESULT_DIR = "results"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# =========================
# BEST PARAM PATH
# =========================
def get_best_param_path(mode):

    return f"best_params_{mode}.txt"


# =========================
# LOAD PARAMS
# =========================
def load_best_params(mode):

    best_param_path = get_best_param_path(
        mode
    )

    default_params = {

        "lr":
            1e-4,

        "batch_size":
            16,

        "weight_decay":
            1e-4,

        "depth":
            DEPTH,

        "mode":
            mode
    }

    # =========================
    # FILE NOT FOUND
    # =========================
    if not os.path.exists(
        best_param_path
    ):

        print(
            f"\n⚠ {best_param_path} not found"
        )

        print(
            "Using default parameters"
        )

        return default_params

    # =========================
    # TRY LOAD
    # =========================
    try:

        with open(
            best_param_path,
            "r"
        ) as f:

            loaded_params = ast.literal_eval(
                f.read()
            )

        # =========================
        # MERGE DEFAULT + LOADED
        # =========================
        default_params.update(
            loaded_params
        )

        params = default_params

        print(
            f"\n✅ Loaded tuned parameters for {mode}"
        )

        print(params)

        return params

    # =========================
    # LOAD FAILED
    # =========================
    except Exception as e:

        print(
            "\n⚠ Failed to load tuning params"
        )

        print(e)

        print(
            "\nUsing default parameters"
        )

        return default_params


# =========================
# TRAIN
# =========================
def train_one_epoch(

    model,
    loader,
    optimizer,
    criterion,
    DEVICE
):

    model.train()

    total_loss = 0

    loop = tqdm(
        loader,
        leave=False,
        desc="Train"
    )

    for imgs, labels in loop:

        imgs = imgs.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(

            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(loader)


# =========================
# VALIDATE
# =========================
def validate(

    model,
    loader,
    criterion,
    DEVICE
):

    model.eval()

    total_loss = 0

    all_preds = []

    all_labels = []

    all_probs = []

    loop = tqdm(

        loader,

        leave=False,

        desc="Val"
    )

    with torch.no_grad():

        for imgs, labels in loop:

            imgs = imgs.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(imgs)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            preds = outputs.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            loop.set_postfix(

                loss=f"{loss.item():.4f}"
            )

    all_preds = np.array(
        all_preds
    )

    all_labels = np.array(
        all_labels
    )

    acc = (
        all_preds == all_labels
    ).mean()

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs
        )

    except:

        auc = 0.0

    return {

        "loss":
            total_loss / len(loader),

        "acc":
            acc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "auc":
            auc
    }


# =========================
# CREATE DATASET
# =========================
def create_dataset(

    train_df,
    val_df,
    mode,
    params
):

    # =========================
    # 3D
    # =========================
    if mode == "3d":

        train_dataset = Dataset3D(

            train_df,

            ROOT_DIR,

            depth=params.get(
                "depth",
                DEPTH
            ),

            preload=True
        )

        val_dataset = Dataset3D(

            val_df,

            ROOT_DIR,

            depth=params.get(
                "depth",
                DEPTH
            ),

            preload=True
        )

    # =========================
    # 2D / 2.5D
    # =========================
    else:

        train_dataset = Dataset2D(

            train_df,

            ROOT_DIR,

            mode=mode,

            preload=True
        )

        val_dataset = Dataset2D(

            val_df,

            ROOT_DIR,

            mode=mode,

            preload=True
        )

    return (
        train_dataset,
        val_dataset
    )


# =========================
# CREATE MODEL
# =========================
def create_model(mode):

    if mode == "3d":

        model = ResNet3D()

    else:

        model = ResNet2D()

    return model.to(DEVICE)


# =========================
# TRAIN 1 FOLD
# =========================
def train_fold(

    train_csv,
    val_csv,
    params,
    fold_idx,
    mode
):

    train_df = pd.read_csv(
        train_csv
    )

    val_df = pd.read_csv(
        val_csv
    )

    train_dataset, val_dataset = create_dataset(

        train_df,
        val_df,
        mode,
        params
    )

    # =========================
    # BATCH SIZE
    # =========================
    if mode == "3d":

        batch_size = params.get(
            "batch_size",
            2
        )

    else:

        batch_size = params.get(
            "batch_size",
            16
        )

    # =========================
    # DATALOADER
    # =========================
    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=True,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True,
    )

    # =========================
    # MODEL
    # =========================
    model = create_model(mode)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(

        model.parameters(),

        lr=params.get(
            "lr",
            1e-4
        ),

        weight_decay=params.get(
            "weight_decay",
            1e-4
        )
    )

    best_metrics = None

    best_acc = 0

    # =========================
    # HISTORY
    # =========================
    history = []

    epoch_bar = tqdm(

        range(EPOCHS),

        desc=f"{mode} Fold {fold_idx}",

        leave=True
    )

    for epoch in epoch_bar:

        train_loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            DEVICE
        )

        val_metrics = validate(

            model,

            val_loader,

            criterion,

            DEVICE
        )

        # =========================
        # SAVE HISTORY
        # =========================
        history.append({

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "val_loss":
                val_metrics["loss"],

            "acc":
                val_metrics["acc"],

            "precision":
                val_metrics["precision"],

            "recall":
                val_metrics["recall"],

            "f1":
                val_metrics["f1"],

            "auc":
                val_metrics["auc"]
        })

        epoch_bar.set_postfix({

            "loss":
                f"{val_metrics['loss']:.4f}",

            "acc":
                f"{val_metrics['acc']:.4f}",

            "f1":
                f"{val_metrics['f1']:.4f}"
        })

        print(

            f"[{mode}] "
            f"[Fold {fold_idx}] "
            f"Epoch {epoch+1} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "

            f"Acc: "
            f"{val_metrics['acc']:.4f} | "

            f"F1: "
            f"{val_metrics['f1']:.4f} | "

            f"AUC: "
            f"{val_metrics['auc']:.4f}"
        )

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_metrics["acc"] > best_acc:

            best_acc = val_metrics["acc"]

            best_metrics = val_metrics

            model_path = os.path.join(

                MODEL_DIR,

                f"best_model_fold_{fold_idx}_{mode}.pth"
            )

            torch.save(

                model.state_dict(),

                model_path
            )

            print(
                f"\n✅ Model saved: {model_path}"
            )

    # =========================
    # SAVE HISTORY CSV
    # =========================
    history_df = pd.DataFrame(
        history
    )

    history_path = os.path.join(

        RESULT_DIR,

        f"history_fold_{fold_idx}_{mode}.csv"
    )

    history_df.to_csv(

        history_path,

        index=False
    )

    print(
        f"\n✅ History saved: {history_path}"
    )

    return best_metrics


# =========================
# K-FOLD
# =========================
def run_kfold(mode):

    params = load_best_params(
        mode
    )

    print(
        f"\n========== MODE: {mode} =========="
    )

    results = []

    for i in range(1, 6):

        train_csv = os.path.join(

            KFOLD_DIR,

            f"fold_{i}_train.csv"
        )

        val_csv = os.path.join(

            KFOLD_DIR,

            f"fold_{i}_val.csv"
        )

        metrics = train_fold(

            train_csv,

            val_csv,

            params,

            i,

            mode
        )

        metrics["fold"] = i

        results.append(metrics)

    df = pd.DataFrame(results)

    save_path = os.path.join(

        RESULT_DIR,

        f"kfold_results_{mode}.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        "\n=== K-FOLD RESULT ==="
    )

    print(df)

    print("\nMEAN:")

    print(df.mean())

    print(
        f"\n✅ KFold result saved: {save_path}"
    )


# =========================
# TEST
# =========================
def run_test(mode):

    params = load_best_params(
        mode
    )

    test_df = pd.read_csv(

        os.path.join(

            SPLIT_DIR,

            "test.csv"
        )
    )

    # =========================
    # DATASET
    # =========================
    if mode == "3d":

        dataset = Dataset3D(

            test_df,

            ROOT_DIR,

            depth=params.get(
                "depth",
                DEPTH
            ),

            preload=True
        )

        batch_size = params.get(
            "batch_size",
            2
        )

    else:

        dataset = Dataset2D(

            test_df,

            ROOT_DIR,

            mode=mode,

            preload=True
        )

        batch_size = params.get(
            "batch_size",
            16
        )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================
    model = create_model(mode)

    model_path = os.path.join(

        MODEL_DIR,

        f"best_model_fold_1_{mode}.pth"
    )

    model.load_state_dict(

        torch.load(
            model_path
        )
    )

    criterion = nn.CrossEntropyLoss()

    metrics = validate(

        model,

        loader,

        criterion,

        DEVICE
    )

    print(
        f"\n=== FINAL TEST {mode} ==="
    )

    print(metrics)

    # =========================
    # SAVE FINAL TEST
    # =========================
    test_result_df = pd.DataFrame(
        [metrics]
    )

    test_save_path = os.path.join(

        RESULT_DIR,

        f"final_test_{mode}.csv"
    )

    test_result_df.to_csv(

        test_save_path,

        index=False
    )

    print(
        f"\n✅ Final test saved: {test_save_path}"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    for mode in MODES:

        print(
            "\n=============================="
        )

        print(
            f"TRAINING MODE: {mode}"
        )

        print(
            "=============================="
        )

        run_kfold(mode)

        print(
            f"\n=== TEST MODE {mode} ==="
        )

        run_test(mode)