import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


class DataAugmentor:
    def __init__(self, img_size=224, aug_count=2):
        self.img_size = img_size
        self.aug_count = aug_count

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        ])

        self.resize = transforms.Resize((img_size, img_size))

    def augment_image(self, input_path, output_dir, base_name):
        img = Image.open(input_path).convert("RGB")

        augmented_files = []

        for i in range(self.aug_count):
            aug_img = self.augment(img)
            aug_img = self.resize(aug_img)

            filename = f"{base_name}_aug{i}.jpg"
            save_path = os.path.join(output_dir, filename)

            aug_img.save(save_path)
            augmented_files.append(filename)

        return augmented_files

    def augment_from_csv(self, csv_path, input_dir, output_dir):
        df = pd.read_csv(csv_path)

        os.makedirs(output_dir, exist_ok=True)

        new_records = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row["filename"]
            label = row["label"]

            input_path = os.path.join(input_dir, filename)

            base_name = os.path.splitext(filename)[0]

            # Copy original (penting!)
            original_output = os.path.join(output_dir, filename)
            Image.open(input_path).convert("RGB").save(original_output)

            new_records.append({
                "filename": filename,
                "label": label
            })

            # Augment
            aug_files = self.augment_image(input_path, output_dir, base_name)

            for aug_file in aug_files:
                new_records.append({
                    "filename": aug_file,
                    "label": label
                })

        # Save mapping baru
        new_df = pd.DataFrame(new_records)
        new_df.to_csv(os.path.join(output_dir, "augmented_mapping.csv"), index=False)

        print("Augmentation selesai!")

aug = DataAugmentor(aug_count=2)

aug.augment_from_csv(
    csv_path="Dataset/Processed/processed_mapping.csv",
    input_dir="Dataset/Processed",
    output_dir="Dataset/Augmented"
)