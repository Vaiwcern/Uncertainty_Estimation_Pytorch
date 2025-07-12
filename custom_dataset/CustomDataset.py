import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import imageio.v2 as imageio
import cv2
import random
import pandas as pd
from sklearn.utils import shuffle

class RTDataset(Dataset):
    def __init__(self, dataset_dir, add_channel=True, normalize=True, train=True, thin_label=False):
        self.image_dir = Path(dataset_dir) / ("imagery" if train else "imagery_test")
        self.mask_dir = Path(dataset_dir) / ("masks" if thin_label else "masks_thick")
        self.normalize = normalize
        self.augment = train
        self.add_channel = add_channel

        print("📂 Looking for images in:", self.image_dir.resolve())
        self.image_files = sorted(self.image_dir.glob("*.png"))
        print("📸 Found:", len(self.image_files), "images")

        self.mask_files = [
            self.mask_dir / f"{'_'.join(f.stem.split('_')[:-4])}_osm_{'_'.join(f.stem.split('_')[4:])}.png"
            for f in self.image_files
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_path = str(self.image_files[idx])
            mask_path = str(self.mask_files[idx])

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            image = imageio.imread(image_path)
            mask = imageio.imread(mask_path)[:, :, 0]

            mask = (mask >= 128).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            if self.normalize:
                image = image / 255.0

            if self.augment:
                image, mask = self.augment_pair(image, mask)

            if self.add_channel:
                zero_channel = np.zeros_like(image[..., :1])
                image = np.concatenate([image, zero_channel], axis=-1)

            image = torch.tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
            mask = torch.tensor(mask.transpose(2, 0, 1).copy(), dtype=torch.float32)

            filename = os.path.basename(image_path)
            return image, mask, filename

        except Exception as e:
            print(f"[❌ ERROR] idx={idx}, image={image_path}, mask={mask_path}, error={e}")
            raise e

    def augment_pair(self, image, mask):
        if random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        if random.random() < 0.5:
            angle = random.uniform(-180, 180)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.squeeze(), matrix, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
        return image, mask

class MassachusettsDataset(Dataset):
    def __init__(self, dataset_dir, split='train', batch_size=8, normalize=True, add_channel=True):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment = (split == 'train')
        self.add_channel = add_channel

        df = pd.read_csv(self.dataset_dir / "metadata.csv")
        df = df[df['split'] == split]

        self.image_files = [self.dataset_dir / p for p in df['tiff_image_path']]
        self.mask_files = [self.dataset_dir / p for p in df['tif_label_path']]
        self.filenames = [Path(p).name for p in df['tiff_image_path']]

        if self.augment:
            self.image_files, self.mask_files, self.filenames = shuffle(
                self.image_files, self.mask_files, self.filenames, random_state=42
            )

        print("📂 Looking for images in:", self.dataset_dir.resolve())
        print("📄 Using metadata.csv split:", split)
        print("📸 Found:", len(self.image_files), "images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_path = str(self.image_files[idx])
            mask_path = str(self.mask_files[idx])

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            image = imageio.imread(image_path)
            mask = imageio.imread(mask_path)

            # Ensure mask is binary and single-channel
            mask = (mask >= 128).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            if self.normalize:
                image = image / 255.0

            if self.augment:
                image, mask = self.augment_pair(image, mask)

            if self.add_channel:
                zero_channel = np.zeros_like(image[..., :1])
                image = np.concatenate([image, zero_channel], axis=-1)

            image = torch.tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
            mask = torch.tensor(mask.transpose(2, 0, 1).copy(), dtype=torch.float32)

            filename = os.path.basename(image_path)
            return image, mask, filename

        except Exception as e:
            print(f"[❌ ERROR] idx={idx}, image={image_path}, mask={mask_path}, error={e}")
            raise e

    def augment_pair(self, image, mask):
        if random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        if random.random() < 0.5:
            angle = random.uniform(-180, 180)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.squeeze(), matrix, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
        return image, mask
