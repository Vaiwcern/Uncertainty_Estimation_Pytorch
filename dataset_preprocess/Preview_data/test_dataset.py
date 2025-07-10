import os
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from custom_dataset.CustomDataset import RTDataset  # đảm bảo file RTDataset của bạn được lưu là `rtdataset.py`

def save_image_and_mask(image, mask, filename_prefix, save_dir):
    image = image.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    mask = mask.numpy().squeeze()            # (1, H, W) -> (H, W)

    image_vis = image[..., :3]               # Chỉ hiển thị RGB nếu có kênh phụ
    mask_vis = np.stack([mask] * 3, axis=-1)  # Đổi mask thành 3 channel để hiển thị cạnh ảnh

    combined = np.concatenate([image_vis, mask_vis], axis=1)

    plt.imsave(os.path.join(save_dir, f"{filename_prefix}.png"), combined)

def debug_dataset(dataset, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (image, mask) in enumerate(loader):
        if i >= 5: break
        save_image_and_mask(image[0], mask[0], f"{prefix}_{i}", save_dir)

if __name__ == "__main__":
    dataset_path = "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512"  # TODO: chỉnh path đúng

    train_dataset = RTDataset(dataset_path, add_channel=False, normalize=True, thin_label=False, train=True)
    test_dataset = RTDataset(dataset_path, add_channel=False, normalize=True, thin_label=False, train=False)

    debug_dataset(train_dataset, "debug_output/train", "train")
    debug_dataset(test_dataset, "debug_output/test", "test")

    print("✅ Done. Check 'debug_output/train' and 'debug_output/test' folders.")
