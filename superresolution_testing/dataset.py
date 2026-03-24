import numpy as np
import torch
from torch.utils.data import Dataset
import os
import augmentation
import matplotlib.pyplot as plt


class PatchDataset(Dataset):
    def __init__(self, hr_dir, if_translation=True, if_rotation=True, if_flip=True, if_downsample=True, flip_prob=0.5, tgt_patch_size=(48, 48, 48), curr_patch_size=(64, 64, 64)):
        self.hr_dir = hr_dir
        # store full patch paths
        self.hr_paths = []
        # store corresponding image names
        self.hr_img_names = []

        for image_folder in os.listdir(hr_dir):
            image_path = os.path.join(hr_dir, image_folder)

            # Ensure it's a directory
            if os.path.isdir(image_path):
                for patch in os.listdir(image_path):
                    patch_path = os.path.join(image_path, patch)

                    if os.path.isfile(patch_path):
                        self.hr_paths.append(patch_path)
                        self.hr_img_names.append(image_folder)
        
        # augmentation args
        self.translation = if_translation
        self.rotation = if_rotation
        self.flip = if_flip
        self.downsample = if_downsample
        self.flip_prob = flip_prob
        self.tgt_patch_size = tgt_patch_size
        self.curr_patch_size = curr_patch_size

    def __len__(self):
        return len(self.hr_paths)

    def spatial_augment(self, patch):
        """Apply spatial augmentations (translation, rotation, flip) that must be shared between HR and LR."""
        if self.translation:
            patch = augmentation.translation(patch, self.tgt_patch_size, self.curr_patch_size)
        if self.rotation:
            patch = augmentation.rotation(patch)
        if self.flip:
            patch = augmentation.flip(patch, self.flip_prob)
        return patch

    def __getitem__(self, index):
        hr_patch = np.load(self.hr_paths[index])

        # Apply spatial augmentations to get the HR patch
        hr_patch = self.spatial_augment(hr_patch)

        # Create LR patch by downsampling the spatially-augmented HR patch
        if self.downsample:
            lr_patch = augmentation.downsample(hr_patch)
        else:
            lr_patch = hr_patch.copy()

        return torch.tensor(hr_patch, dtype=torch.float32), torch.tensor(lr_patch, dtype=torch.float32)

