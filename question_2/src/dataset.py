import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CityDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname))

        if self.transform:
            img = self.transform(img)

        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


def get_splits(img_dir, seed=42):
    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    random.seed(seed)
    random.shuffle(files)

    split = int(0.8 * len(files))
    return files[:split], files[split:]
