import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class LIDCNoduleDataset(Dataset):
    def __init__(self, df, root_dir, mode="2d"):
        """
        mode: '2d' atau '2.5d'
        """
        self.df = df
        self.root_dir = root_dir
        self.mode = mode

        self.samples = []

        for idx, row in self.df.iterrows():

         
            patient_id = row["patient_id"].split(".")[-1]
            sample_id = str(row["sample_id"]).zfill(3)
            lidc_id = str(row["lidc_id"])
            label = str(row["label"])

            sample_path = os.path.join(
                root_dir,
                "Nodule_Images",
                label,
                lidc_id,
                sample_id
            )

            if not os.path.exists(sample_path):
                continue

            self.samples.append({
                "path": sample_path,
                "label": 1 if label == "malignant" else 0,
                "z": float(row["z_position"])
            })


    def __len__(self):
        return len(self.samples)


    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        img = cv2.resize(img, (224, 224))
        return img


    def __getitem__(self, idx):

        sample = self.samples[idx]
        folder = sample["path"]
        label = sample["label"]
        target_z = int(round(sample["z"]))

        # =========================
        # Get all slices in folder
        # =========================
        files = [f for f in os.listdir(folder) if f.endswith(".png")]

        # parse z dari filename
        z_files = []

        for f in files:
            try:
                z_val = int(f.replace(".png", ""))
                z_files.append((z_val, f))
            except:
                continue

        # sort berdasarkan z
        z_files = sorted(z_files, key=lambda x: x[0])

        # =========================
        # Cari slice tengah (closest ke z CSV)
        # =========================
        closest_idx = min(
            range(len(z_files)),
            key=lambda i: abs(z_files[i][0] - target_z)
        )

        # =========================
        # MODE 2D
        # =========================
        if self.mode == "2d":
            z_val, filename = z_files[closest_idx]
            img = self.load_image(os.path.join(folder, filename))

            if img is None:
                return self.__getitem__((idx + 1) % len(self))

            # jadi 3 channel
            img = np.stack([img]*3, axis=0)

        # =========================
        # MODE 2.5D
        # =========================
        else:
            if len(z_files) < 3:
                return self.__getitem__((idx + 1) % len(self))

            # robust selection
            if closest_idx == 0:
                indices = [0, 1, 2]
            elif closest_idx == len(z_files) - 1:
                indices = [
                    len(z_files) - 3,
                    len(z_files) - 2,
                    len(z_files) - 1
                ]
            else:
                indices = [closest_idx - 1, closest_idx, closest_idx + 1]

            imgs = []

            for i in indices:
                z_val, filename = z_files[i]
                img = self.load_image(os.path.join(folder, filename))

                if img is None:
                    return self.__getitem__((idx + 1) % len(self))

                imgs.append(img)

            img = np.stack(imgs, axis=0)  # (3, H, W)

        # =========================
        # Normalize
        # =========================
        img = img.astype(np.float32) / 255.0

        img = torch.tensor(img)
        label = torch.tensor(label).long()

        return img, label