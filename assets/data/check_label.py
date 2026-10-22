
import os
import cv2
import numpy as np

label_dir = "/home/SPG-CD/assets/data/train/label"

all_zero_files = []
has_change_files = []
read_failed_files = []

for fname in sorted(os.listdir(label_dir)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        file_path = os.path.join(label_dir, fname)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARNING] Failed to read: {fname}")
            read_failed_files.append(fname)
            continue

        # Check whether all pixels in the label image are zero.
        if np.all(img == 0):
            all_zero_files.append(fname)
        else:
            has_change_files.append((fname, int(img.max())))

print("=" * 60)
print(f"Label directory: {label_dir}")
print(f"Number of all-zero (no-change) labels: {len(all_zero_files)}")
print(f"Number of labels containing non-zero pixels: {len(has_change_files)}")
print(f"Number of unreadable files: {len(read_failed_files)}")
print("=" * 60)

if all_zero_files:
    print("\nFiles with all-zero labels:")
    for name in all_zero_files:
        print(f" - {name}")

if has_change_files:
    print("\nFiles containing non-zero label pixels:")
    for name, max_val in has_change_files:
        print(f" - {name} | Maximum pixel value: {max_val}")

if read_failed_files:
    print("\nUnreadable files:")
    for name in read_failed_files:
        print(f" - {name}")
