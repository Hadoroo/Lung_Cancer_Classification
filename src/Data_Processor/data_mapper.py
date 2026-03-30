import os
import xml.etree.ElementTree as ET
import pandas as pd


DATASET_PATH = "Dataset"
OUTPUT = "Dataset_Processed/lidc_nodules.csv"


def malignancy_to_label(score):

    score = int(score)

    if score <= 2.5:
        return "benign"

    if score >= 3.5:
        return "malignant"

    return None


rows = []


for root_dir, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(root_dir, file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except:
            continue


        patient_id = None

        for elem in root.iter():
            if "SeriesInstanceUid" in elem.tag:
                patient_id = elem.text


        for nodule in root.iter():

            if "unblindedReadNodule" not in nodule.tag:
                continue


            malignancy = None

            for elem in nodule.iter():
                if "malignancy" in elem.tag:
                    malignancy = elem.text


            if malignancy is None:
                continue


            label = malignancy_to_label(malignancy)

            if label is None:
                continue


            for roi in nodule.iter():

                if "roi" not in roi.tag:
                    continue


                z_pos = None
                xs = []
                ys = []

                for elem in roi.iter():

                    if "imageZposition" in elem.tag:
                        z_pos = elem.text

                    if "xCoord" in elem.tag:
                        xs.append(int(elem.text))

                    if "yCoord" in elem.tag:
                        ys.append(int(elem.text))


                if len(xs) == 0:
                    continue


                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)


                rows.append({
                    "patient_id": patient_id,
                    "xml_path": xml_path,
                    "z_position": z_pos,
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "malignancy_score": malignancy,
                    "label": label
                })


df = pd.DataFrame(rows)

df.to_csv(OUTPUT, index=False)

print("Total nodules:", len(df))
print("CSV saved:", OUTPUT)