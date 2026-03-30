import pydicom
import matplotlib.pyplot as plt

# load dicom
dicom_path = r"Dataset\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-118.dcm"
ds = pydicom.dcmread(dicom_path)

# ambil pixel image
image = ds.pixel_array

# tampilkan
plt.imshow(image, cmap="gray")
plt.title("CT Scan")
plt.axis("off")
plt.show()