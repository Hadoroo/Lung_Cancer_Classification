import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from tqdm import tqdm


class Dataset2D(Dataset):

    def __init__(
        self,
        df,
        root_dir,
        mode="2d",
        preload=True
    ):

        self.df = df
        self.root_dir = root_dir
        self.mode = mode
        self.preload = preload

        self.samples = []

        # =========================
        # RAM CACHE
        # =========================
        self.cache = {}

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

        # =========================
        # PRELOAD TO RAM
        # =========================
        if self.preload:

            print(f"\n🔥 Preloading {mode} dataset to RAM...")

            for sample in tqdm(self.samples):

                folder = sample["path"]

                files = [
                    f for f in os.listdir(folder)
                    if f.endswith(".png")
                ]

                for f in files:

                    full_path = os.path.join(folder, f)

                    if full_path in self.cache:
                        continue

                    img = cv2.imread(
                        full_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is None:
                        continue

                    img = cv2.resize(img, (224,224))

                    self.cache[full_path] = img

            print(
                f"✅ Cached {len(self.cache)} images in RAM"
            )


    def __len__(self):
        return len(self.samples)


    def load_image(self, path):

        # =========================
        # FROM RAM
        # =========================
        if self.preload:

            img = self.cache.get(path)

            if img is None:
                return None

            return img

        # =========================
        # FROM DISK
        # =========================
        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            return None

        img = cv2.resize(img, (224,224))

        return img


    def __getitem__(self, idx):

        sample = self.samples[idx]

        folder = sample["path"]

        label = sample["label"]

        target_z = int(round(sample["z"]))

        # =========================
        # GET FILES
        # =========================
        files = [
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ]

        z_files = []

        for f in files:

            try:

                z_val = int(f.replace(".png", ""))

                z_files.append((z_val, f))

            except:
                continue

        z_files = sorted(
            z_files,
            key=lambda x: x[0]
        )

        if len(z_files) == 0:
            return self.__getitem__(
                (idx + 1) % len(self)
            )

        # =========================
        # CENTER SLICE
        # =========================
        closest_idx = min(
            range(len(z_files)),
            key=lambda i: abs(
                z_files[i][0] - target_z
            )
        )

        # =========================
        # MODE 2D
        # =========================
        if self.mode == "2d":

            z_val, filename = z_files[closest_idx]

            full_path = os.path.join(
                folder,
                filename
            )

            img = self.load_image(full_path)

            if img is None:
                return self.__getitem__(
                    (idx + 1) % len(self)
                )

            # grayscale -> 3 channel
            img = np.stack([img]*3, axis=0)

        # =========================
        # MODE 2.5D
        # =========================
        else:

            if len(z_files) < 3:
                return self.__getitem__(
                    (idx + 1) % len(self)
                )

            if closest_idx == 0:

                indices = [0,1,2]

            elif closest_idx == len(z_files)-1:

                indices = [
                    len(z_files)-3,
                    len(z_files)-2,
                    len(z_files)-1
                ]

            else:

                indices = [
                    closest_idx-1,
                    closest_idx,
                    closest_idx+1
                ]

            imgs = []

            for i in indices:

                z_val, filename = z_files[i]

                full_path = os.path.join(
                    folder,
                    filename
                )

                img = self.load_image(full_path)

                if img is None:
                    return self.__getitem__(
                        (idx + 1) % len(self)
                    )

                imgs.append(img)

            img = np.stack(imgs, axis=0)

        # =========================
        # NORMALIZE
        # =========================
        img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).float()

        label = torch.tensor(label).long()

        return img, label
    
    
class Dataset3D(Dataset):

    def __init__(
        self,
        df,
        root_dir,
        depth=16,
        preload=True
    ):

        self.df = df
        self.root_dir = root_dir
        self.depth = depth
        self.preload = preload

        self.samples = []

        # =========================
        # RAM CACHE
        # =========================
        self.cache = {}

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

        # =========================
        # PRELOAD TO RAM
        # =========================
        if self.preload:

            print("\n🔥 Preloading images to RAM...")

            for sample in tqdm(self.samples):

                folder = sample["path"]

                files = [
                    f for f in os.listdir(folder)
                    if f.endswith(".png")
                ]

                for f in files:

                    full_path = os.path.join(folder, f)

                    if full_path in self.cache:
                        continue

                    img = cv2.imread(
                        full_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is None:
                        continue

                    img = cv2.resize(img, (224,224))

                    self.cache[full_path] = img

            print(
                f"✅ Cached {len(self.cache)} images in RAM"
            )


    def __len__(self):
        return len(self.samples)


    def load_image(self, path):

        # =========================
        # FROM RAM
        # =========================
        if self.preload:

            img = self.cache.get(path)

            if img is None:
                return None

            return img

        # =========================
        # FROM DISK
        # =========================
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        img = cv2.resize(img, (224,224))

        return img


    def __getitem__(self, idx):

        sample = self.samples[idx]

        folder = sample["path"]

        label = sample["label"]

        target_z = int(round(sample["z"]))

        # =========================
        # GET FILES
        # =========================
        files = [
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ]

        z_files = []

        for f in files:

            try:

                z_val = int(f.replace(".png", ""))

                z_files.append((z_val, f))

            except:
                continue

        z_files = sorted(z_files, key=lambda x: x[0])

        if len(z_files) == 0:
            return self.__getitem__(
                (idx + 1) % len(self)
            )

        # =========================
        # CENTER SLICE
        # =========================
        closest_idx = min(
            range(len(z_files)),
            key=lambda i: abs(
                z_files[i][0] - target_z
            )
        )

        # =========================
        # DEPTH WINDOW
        # =========================
        half_depth = self.depth // 2

        start_idx = closest_idx - half_depth
        end_idx   = closest_idx + half_depth

        if start_idx < 0:
            start_idx = 0
            end_idx = self.depth

        if end_idx > len(z_files):
            end_idx = len(z_files)
            start_idx = max(
                0,
                end_idx - self.depth
            )

        selected = z_files[start_idx:end_idx]

        # =========================
        # LOAD VOLUME
        # =========================
        volume = []

        for z_val, filename in selected:

            full_path = os.path.join(
                folder,
                filename
            )

            img = self.load_image(full_path)

            if img is None:
                return self.__getitem__(
                    (idx + 1) % len(self)
                )

            volume.append(img)

        # =========================
        # PAD IF NEEDED
        # =========================
        while len(volume) < self.depth:

            volume.append(
                np.zeros(
                    (224,224),
                    dtype=np.uint8
                )
            )

        volume = np.stack(volume, axis=0)

        volume = volume.astype(np.float32) / 255.0

        volume = torch.tensor(volume).float()

        # [D,H,W] -> [1,D,H,W]
        volume = volume.unsqueeze(0)

        label = torch.tensor(label).long()

        return volume, label