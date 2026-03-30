import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

CSV_PATH = "Dataset_Processed/lidc_nodules_final.csv"
SAVE_DIR = "Dataset_Processed/splits"
KFOLD_DIR = os.path.join(SAVE_DIR, "kfold")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(KFOLD_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

# ambil patient id
df["patient"] = df["patient_id"].apply(lambda x: x.split(".")[-1])

# =========================
# STEP 1: TRAIN / TEST SPLIT
# =========================
patient_df = df.groupby("patient")["label"].first().reset_index()

trainval_p, test_p = train_test_split(
    patient_df,
    test_size=0.2,
    stratify=patient_df["label"],
    random_state=42
)

trainval_df = df[df["patient"].isin(trainval_p["patient"])].reset_index(drop=True)
test_df     = df[df["patient"].isin(test_p["patient"])].reset_index(drop=True)

# =========================
# STEP 2: TRAIN / VAL SPLIT
# =========================
train_p, val_p = train_test_split(
    trainval_p,
    test_size=0.1,
    stratify=trainval_p["label"],
    random_state=42
)

train_df = df[df["patient"].isin(train_p["patient"])].reset_index(drop=True)
val_df   = df[df["patient"].isin(val_p["patient"])].reset_index(drop=True)

# =========================
# SAVE BASE SPLIT
# =========================
train_df.to_csv(os.path.join(SAVE_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(SAVE_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(SAVE_DIR, "test.csv"), index=False)

print("Train/Val/Test saved!")

# =========================
# STEP 3: K-FOLD (TRAIN + VAL)
# =========================
trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)

patient_df = trainval_df.groupby("patient")["label"].first().reset_index()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(patient_df["patient"], patient_df["label"])
):

    train_patients = patient_df.iloc[train_idx]["patient"]
    val_patients   = patient_df.iloc[val_idx]["patient"]

    fold_train_df = trainval_df[
        trainval_df["patient"].isin(train_patients)
    ].reset_index(drop=True)

    fold_val_df = trainval_df[
        trainval_df["patient"].isin(val_patients)
    ].reset_index(drop=True)

    fold_train_df.to_csv(
        os.path.join(KFOLD_DIR, f"fold_{fold+1}_train.csv"),
        index=False
    )

    fold_val_df.to_csv(
        os.path.join(KFOLD_DIR, f"fold_{fold+1}_val.csv"),
        index=False
    )

    print(f"Fold {fold+1} saved!")

print("All splits done!")