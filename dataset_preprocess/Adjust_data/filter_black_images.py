import os
from pathlib import Path
import imageio
import numpy as np
from tqdm import tqdm

def count_black_masks(dataset_root, thin_label=False):
    dataset_root = Path(dataset_root)
    imagery_dir = dataset_root / "imagery"
    mask_dir = dataset_root / ("masks" if thin_label else "masks_thick")

    image_files = sorted(imagery_dir.glob("*.png"))
    print(f"🔍 Found {len(image_files)} images. Checking for empty masks...")

    black_count = 0
    black_samples = []

    for img_path in tqdm(image_files):
        parts = img_path.stem.split('_')
        mask_name = f"{'_'.join(parts[:-4])}_osm_{'_'.join(parts[4:])}.png"
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            continue

        try:
            mask = imageio.imread(mask_path)[:, :, 0]
            if np.sum(mask >= 128) == 0:
                black_count += 1
                black_samples.append((img_path, mask_path))
        except Exception as e:
            print(f"❌ Failed to read {mask_path}: {e}")
            continue

    print(f"🧹 Found {black_count} fully black masks.")

    # ❗Optional removal — uncomment below to delete the files
    # for img_path, mask_path in black_samples:
    #     img_path.unlink(missing_ok=True)
    #     mask_path.unlink(missing_ok=True)

# Example usage
count_black_masks("/home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_512", thin_label=False)
