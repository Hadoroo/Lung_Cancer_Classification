import os
import pandas as pd
import pydicom
import cv2
import numpy as np
from tqdm import tqdm

CSV_PATH = r"Dataset_Processed\lidc_nodules.csv"
OUTPUT_ROOT = r"Dataset_Processed"
NEW_CSV_PATH = r"Dataset_Processed\lidc_nodules_final.csv"

df = pd.read_csv(CSV_PATH)

# tambah kolom baru
df["lidc_id"] = ""
df["sample_id"] = ""

sample_counter = {}

def dicom_to_hu(dcm):
    img = dcm.pixel_array.astype(np.int16)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    slope = getattr(dcm, "RescaleSlope", 1)

    if slope != 1:
        img = slope * img

    img += np.int16(intercept)
    return img


def lung_window(img, center=-600, width=1500):
    min_val = center - width // 2
    max_val = center + width // 2

    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    return (img * 255).astype(np.uint8)


def get_lidc_id(series_folder):
    name = series_folder.split(os.sep)[-3]
    return name.replace("LIDC-IDRI-", "LIDC_")


df["series_folder"] = df["xml_path"].apply(lambda x: os.path.dirname(x))
series_groups = df.groupby("series_folder")


for series_folder, group in tqdm(series_groups):

    dcm_files = [f for f in os.listdir(series_folder) if f.endswith(".dcm")]
    slice_map = []

    for dcm_file in dcm_files:
        dcm_path = os.path.join(series_folder, dcm_file)

        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        except:
            continue

        if not hasattr(dcm, "ImagePositionPatient"):
            continue

        z = float(dcm.ImagePositionPatient[2])
        slice_map.append((dcm_file, z))

    if len(slice_map) < 3:
        continue

    slice_map = sorted(slice_map, key=lambda x: x[1])
    lidc_id = get_lidc_id(series_folder)

    # counter per patient
    if lidc_id not in sample_counter:
        sample_counter[lidc_id] = 0

    for idx_row, row in group.iterrows():

        xmin, xmax = int(row["xmin"]), int(row["xmax"])
        ymin, ymax = int(row["ymin"]), int(row["ymax"])
        target_z = float(row["z_position"])
        label = row["label"]

        # assign sample_id
        sample_id = f"{sample_counter[lidc_id]:03d}"
        sample_counter[lidc_id] += 1

        df.loc[idx_row, "lidc_id"] = lidc_id
        df.loc[idx_row, "sample_id"] = sample_id

        # cari slice tengah
        closest_idx = min(
            range(len(slice_map)),
            key=lambda i: abs(slice_map[i][1] - target_z)
        )

        if closest_idx == 0:
            indices = [0, 1, 2]
        elif closest_idx == len(slice_map) - 1:
            indices = [-3, -2, -1]
        else:
            indices = [closest_idx - 1, closest_idx, closest_idx + 1]

        nodule_folder = os.path.join(
            OUTPUT_ROOT, "Nodule_Images", label, lidc_id, sample_id
        )
        os.makedirs(nodule_folder, exist_ok=True)

        for i, idx_slice in enumerate(indices):

            dcm_file, _ = slice_map[idx_slice]
            dcm_path = os.path.join(series_folder, dcm_file)

            try:
                dcm = pydicom.dcmread(dcm_path)
            except:
                continue

            img = lung_window(dicom_to_hu(dcm))

            h, w = img.shape
            margin = 10

            xmin_m = max(0, xmin - margin)
            xmax_m = min(w, xmax + margin)
            ymin_m = max(0, ymin - margin)
            ymax_m = min(h, ymax + margin)

            roi = img[ymin_m:ymax_m, xmin_m:xmax_m]

            if roi.size == 0:
                continue

            roi = cv2.resize(roi, (224, 224))

            # 🔥 FIX: gunakan index bukan z
            filename = ["-1.png", "0.png", "+1.png"][i]

            cv2.imwrite(os.path.join(nodule_folder, filename), roi)

print("Saving updated CSV...")
df.to_csv(NEW_CSV_PATH, index=False)

print("Done!")