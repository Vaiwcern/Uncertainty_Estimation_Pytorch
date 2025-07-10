import os
import cv2
import numpy as np

def check_image_and_mask_shapes(dataset_dir):
    # Đường dẫn đến các thư mục images và masks trong train và test
    image_dirs = [os.path.join(dataset_dir, "train", "images"), os.path.join(dataset_dir, "test", "images")]
    mask_dirs = [os.path.join(dataset_dir, "train", "masks"), os.path.join(dataset_dir, "test", "masks")]

    for image_dir, mask_dir in zip(image_dirs, mask_dirs):
        # Lấy danh sách các file ảnh và mask
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        if len(image_files) != len(mask_files):
            print(f"❗️ Số lượng ảnh và mask không khớp: {len(image_files)} ảnh và {len(mask_files)} mask trong thư mục {image_dir}.")
            return

        for i in range(len(image_files)):
            image_path = os.path.join(image_dir, image_files[i])
            mask_path = os.path.join(mask_dir, mask_files[i])

            # Đọc ảnh và mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Kiểm tra kích thước ảnh và mask
            if image is None:
                print(f"❌ Ảnh {image_files[i]} không thể đọc.")
            if mask is None:
                print(f"❌ Mask {mask_files[i]} không thể đọc.")

            # In ra shape của ảnh và mask
            print(f"🔍 Ảnh {image_files[i]} có shape: {image.shape}")
            print(f"🔍 Mask {mask_files[i]} có shape: {mask.shape}")

            # Kiểm tra shape ảnh (96x96x3)
            if image.shape != (96, 96, 3):
                print(f"❗️ Ảnh {image_files[i]} có shape không đúng: {image.shape}. Cần là (96, 96, 3)")

            # Kiểm tra shape mask (96x96x1)
            if mask.shape != (96, 96):
                print(f"❗️ Mask {mask_files[i]} có shape không đúng: {mask.shape}. Cần là (96, 96)")

    print("✅ Kiểm tra xong.")

# Đường dẫn tới dataset của bạn
dataset_dir = "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/cell_nuclei_crop_96"

# Kiểm tra các hình ảnh và mask trong dataset
check_image_and_mask_shapes(dataset_dir)
