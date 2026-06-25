import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from Classifier.model import (
    ResNet2D,
    ResNet3D
)

from Data_Loader.data_loader import (
    Dataset2D,
    Dataset3D
)

# ==================================
# CONFIG
# ==================================
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR = "Dataset_Processed"

SPLIT_DIR = (
    "Dataset_Processed/splits"
)

# MODE = "2d"
MODE = "2.5d"
# MODE = "3d"

DEPTH = 16

MODEL_PATH = (
    f"saved_models/"
    f"best_model_fold_4_{MODE}.pth"
)

SAVE_DIR = (
    f"src/Explainable_AI/"
    f"gradcam_results_{MODE}"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

NUM_SAMPLES = 20


# ==================================
# LOAD DATA
# ==================================
test_df = pd.read_csv(

    os.path.join(
        SPLIT_DIR,
        "test.csv"
    )
)

if MODE == "3d":

    dataset = Dataset3D(

        test_df,

        ROOT_DIR,

        depth=DEPTH,

        preload=True
    )

else:

    dataset = Dataset2D(

        test_df,

        ROOT_DIR,

        mode=MODE,

        preload=True
    )

loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False
)


# ==================================
# LOAD MODEL
# ==================================
if MODE == "3d":

    model = ResNet3D()

else:

    model = ResNet2D()

model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


# ==================================
# TARGET LAYER
# ==================================
if MODE == "3d":

    target_layers = [

        model.layer4[-1]
    ]

else:

    target_layers = [

        model.model.layer4[-1]
    ]

cam = GradCAM(

    model=model,

    target_layers=target_layers
)


# ==================================
# GENERATE GRADCAM
# ==================================
for idx, (imgs, labels) in enumerate(loader):

    imgs = imgs.to(DEVICE)

    # ==========================
    # PREDICTION
    # ==========================
    with torch.no_grad():

        outputs = model(imgs)

        probs = torch.softmax(

            outputs,

            dim=1
        )

        pred = outputs.argmax(
            dim=1
        ).item()

        confidence = probs[
            0,
            pred
        ].item()

    # ==========================
    # GENERATE CAM
    # ==========================
    grayscale_cam = cam(
        input_tensor=imgs
    )[0]

    # ==========================
    # IMAGE ARRAY
    # ==========================
    image = imgs[
        0
    ].detach().cpu().numpy()

    # ==================================
    # 2D
    # ==================================
    if MODE == "2d":

        display_img = image[0]

    # ==================================
    # 2.5D
    # ==================================
    elif MODE == "2.5d":

        center_idx = (
            image.shape[0] // 2
        )

        display_img = image[
            center_idx
        ]

    # ==================================
    # 3D
    # ==================================
    else:

        # shape:
        # (D, H, W)
        volume = image[0]

        # ==========================
        # FIND BEST SLICE
        # ==========================
        slice_scores = [

            np.abs(
                volume[i]
            ).sum()

            for i in range(
                volume.shape[0]
            )
        ]

        best_slice_idx = np.argmax(
            slice_scores
        )

        print(
            f"Best slice: "
            f"{best_slice_idx}"
        )

        # ==========================
        # IMAGE SLICE
        # ==========================
        display_img = volume[
            best_slice_idx
        ]

        # ==========================
        # CAM SLICE
        # ==========================
        grayscale_cam = grayscale_cam[
            best_slice_idx
        ]

    # ==========================
    # NORMALIZE IMAGE
    # ==========================
    display_img = (
        display_img -
        display_img.min()
    ) / (
        display_img.max()
        - display_img.min()
        + 1e-8
    )

    # ==========================
    # NORMALIZE CAM
    # ==========================
    grayscale_cam = (
        grayscale_cam -
        grayscale_cam.min()
    ) / (
        grayscale_cam.max()
        - grayscale_cam.min()
        + 1e-8
    )

    # ==========================
    # RGB IMAGE
    # ==========================
    rgb_img = np.stack(

        [display_img] * 3,

        axis=-1
    )

    rgb_img = np.float32(
        rgb_img
    )

    # ==========================
    # VISUALIZATION
    # ==========================
    visualization = show_cam_on_image(

        rgb_img,

        grayscale_cam,

        use_rgb=True
    )

    # ==========================
    # PLOT
    # ==========================
    plt.figure(
        figsize=(10, 4)
    )

    # --------------------------
    # ORIGINAL
    # --------------------------
    plt.subplot(1, 2, 1)

    plt.imshow(
        display_img,
        cmap="gray"
    )

    plt.title(

        f"Original\n"

        f"Label={labels.item()}"
    )

    plt.axis("off")

    # --------------------------
    # GRADCAM
    # --------------------------
    plt.subplot(1, 2, 2)

    plt.imshow(
        visualization
    )

    plt.title(

        f"GradCAM\n"

        f"Pred={pred} "

        f"({confidence:.3f})"
    )

    plt.axis("off")

    plt.tight_layout()

    # ==========================
    # SAVE
    # ==========================
    save_path = os.path.join(

        SAVE_DIR,

        f"sample_{idx}.png"
    )

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    print(
        f"saved -> {save_path}"
    )

    # ==========================
    # LIMIT SAMPLE
    # ==========================
    if idx >= NUM_SAMPLES - 1:

        break