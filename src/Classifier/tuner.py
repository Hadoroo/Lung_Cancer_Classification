import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import pandas as pd
import itertools
import sys
import os

from tqdm import tqdm

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from model import (
    ResNet2D,
    ResNet3D
)

from Data_Loader.data_loader import (
    Dataset2D,
    Dataset3D
)

from trainer import (
    train_one_epoch,
    validate
)

torch.backends.cudnn.benchmark = True

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =========================
# FAST TUNING
# =========================
FAST_TUNING = True

if FAST_TUNING:

    TUNING_FRACTION = 0.2

    EPOCHS = 3

else:

    TUNING_FRACTION = 1.0

    EPOCHS = 10

# =========================
# SEARCH SPACE
# =========================
SEARCH_SPACE = {

    "lr": [

        1e-3,

        1e-4
    ],

    "weight_decay": [

        0,

        1e-4
    ],

    "mode": [

        "2d",

        "2.5d",

        "3d"
    ],

    # khusus 3D
    "depth": [

        8,

        16
    ]
}

ROOT_DIR = "Dataset_Processed"

TRAIN_CSV = (
    "Dataset_Processed/splits/train.csv"
)

VAL_CSV = (
    "Dataset_Processed/splits/val.csv"
)

# =========================
# RUN EXPERIMENT
# =========================
def run_experiment(params):

    print(
        f"\nTesting config: {params}"
    )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    # =========================
    # SAMPLE DATA
    # =========================
    train_df = train_df.sample(

        frac=TUNING_FRACTION,

        random_state=42
    ).reset_index(drop=True)

    val_df = val_df.sample(

        frac=TUNING_FRACTION,

        random_state=42
    ).reset_index(drop=True)

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Val samples: {len(val_df)}"
    )

    mode = params["mode"]

    # =========================
    # DATASET
    # =========================
    if mode in [

        "2d",

        "2.5d"
    ]:

        train_dataset = Dataset2D(

            train_df,

            root_dir=ROOT_DIR,

            mode=mode,

            preload=True
        )

        val_dataset = Dataset2D(

            val_df,

            root_dir=ROOT_DIR,

            mode=mode,

            preload=True
        )

    else:

        train_dataset = Dataset3D(

            train_df,

            root_dir=ROOT_DIR,

            depth=params["depth"],

            preload=True
        )

        val_dataset = Dataset3D(

            val_df,

            root_dir=ROOT_DIR,

            depth=params["depth"],

            preload=True
        )

    # =========================
    # BATCH SIZE
    # =========================
    if mode == "3d":

        batch_size = 2

    else:

        batch_size = 16

    # =========================
    # DATALOADER
    # =========================
    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=2
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=2
    )

    # =========================
    # MODEL
    # =========================
    if mode == "3d":

        model = ResNet3D().to(
            DEVICE
        )

    else:

        model = ResNet2D().to(
            DEVICE
        )

    # =========================
    # LOSS & OPTIMIZER
    # =========================
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

    best_val_acc = 0

    # =========================
    # TRAIN LOOP
    # =========================
    epoch_bar = tqdm(

        range(EPOCHS),

        desc=f"{mode} Epoch",

        leave=False
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

        val_loss = val_metrics["loss"]

        val_acc = val_metrics["acc"]

        epoch_bar.set_postfix({

            "train_loss":
                f"{train_loss:.4f}",

            "val_acc":
                f"{val_acc:.4f}",

            "f1":
                f"{val_metrics['f1']:.4f}"
        })

        print(

            f"[{mode}] "
            f"Epoch {epoch+1} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Val Loss: "
            f"{val_loss:.4f} | "

            f"Acc: "
            f"{val_acc:.4f} | "

            f"F1: "
            f"{val_metrics['f1']:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

    return best_val_acc


# =========================
# GRID SEARCH
# =========================
def grid_search():

    best_results = {}

    # =========================
    # LOOP PER MODE
    # =========================
    for mode in SEARCH_SPACE["mode"]:

        print(
            f"\n======================"
        )

        print(
            f"GRID SEARCH MODE: {mode}"
        )

        print(
            f"======================"
        )

        best_score = 0

        best_params = None

        # =========================
        # PARAM COMBINATION
        # =========================
        if mode == "3d":

            combinations = itertools.product(

                SEARCH_SPACE["lr"],

                SEARCH_SPACE["weight_decay"],

                SEARCH_SPACE["depth"]
            )

        else:

            combinations = itertools.product(

                SEARCH_SPACE["lr"],

                SEARCH_SPACE["weight_decay"]
            )

        # =========================
        # LOOP COMBINATION
        # =========================
        for combo in combinations:

            # =========================
            # BUILD PARAMS
            # =========================
            if mode == "3d":

                params = {

                    "lr":
                        combo[0],

                    "weight_decay":
                        combo[1],

                    "depth":
                        combo[2],

                    "mode":
                        mode,

                    "batch_size":
                        2
                }

            else:

                params = {

                    "lr":
                        combo[0],

                    "weight_decay":
                        combo[1],

                    "mode":
                        mode,

                    "batch_size":
                        16
                }

            score = run_experiment(
                params
            )

            # =========================
            # BEST PARAM
            # =========================
            if score > best_score:

                best_score = score

                best_params = params

                print(
                    "\n🔥 New Best Found!"
                )

                print(
                    best_params
                )

                print(
                    "VAL ACC:",
                    best_score
                )

        best_results[mode] = {

            "params":
                best_params,

            "score":
                best_score
        }

    return best_results


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    best_results = grid_search()

    print(
        "\n======================"
    )

    print(
        "FINAL BEST RESULTS"
    )

    print(
        "======================"
    )

    for mode, result in best_results.items():

        print(
            f"\nMODE: {mode}"
        )

        print(
            "BEST PARAMS:"
        )

        print(
            result["params"]
        )

        print(
            "BEST SCORE:"
        )

        print(
            result["score"]
        )

        # =========================
        # SAVE PER MODE
        # =========================
        save_path = (
            f"best_params_{mode}.txt"
        )

        with open(
            save_path,
            "w"
        ) as f:

            f.write(
                str(
                    result["params"]
                )
            )

        print(
            f"Saved: {save_path}"
        )