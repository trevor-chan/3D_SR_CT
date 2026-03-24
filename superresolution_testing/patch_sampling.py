import os
import numpy as np
import pydicom
from scipy import ndimage
import tqdm
import argparse


# Default config
DEFAULT_CONFIG = {
    "high_res_voxel_size": 0.033,
    "desired_voxel_size": 0.2,
    "patch_size": 64,
    "overlap_size": 32,
    "intensity_threshold": 1000,
}


def get_sorted_dicom_files(path):
    dicom_files = [os.path.join(path, f) for f in os.listdir(path)]
    return sorted(dicom_files)


def load_dicom_block(dicom_files, start, end):
    dicom_slices = [pydicom.dcmread(f).pixel_array for f in dicom_files[start:end]]
    return np.stack(dicom_slices, axis=-1)


def extract_patches(img, patches_path, patch_id, config):
    """Extract patches from a downsampled image block."""
    patch_size = config["patch_size"]
    overlap_size = config["overlap_size"]
    intensity_threshold = config["intensity_threshold"]
    scale = config["high_res_voxel_size"] / config["desired_voxel_size"]

    downsampled_img = ndimage.zoom(img, scale)
    x_size, y_size, z_size = downsampled_img.shape

    if z_size < patch_size:
        print(f"z size is less than patch size: {z_size}")
        return patch_id

    downsampled_img = downsampled_img[:, :, :patch_size]

    for x in range(0, x_size, patch_size - overlap_size):
        for y in range(0, y_size, patch_size - overlap_size):
            if x + patch_size <= x_size and y + patch_size <= y_size:
                patch = downsampled_img[x:x + patch_size, y:y + patch_size, :]
                if np.mean(patch) > intensity_threshold:
                    patch_filename = os.path.join(patches_path, f"{str(patch_id).zfill(8)}.npy")
                    np.save(patch_filename, patch)
                    patch_id += 1

    return patch_id


def process_single_image(image_path, patches_path, config=None):
    """Process a single high-res image folder and extract patches."""
    if config is None:
        config = DEFAULT_CONFIG

    os.makedirs(patches_path, exist_ok=True)

    high_res_files = get_sorted_dicom_files(image_path)
    scale_ratio = config["desired_voxel_size"] / config["high_res_voxel_size"]
    high_res_patch_size = int(config["patch_size"] * scale_ratio + 1)
    high_res_overlap_size = int(config["overlap_size"] * scale_ratio + 1)
    step = high_res_patch_size - high_res_overlap_size

    patch_id = 0
    for i in tqdm.tqdm(range(0, len(high_res_files), step)):
        if i + high_res_patch_size <= len(high_res_files):
            img = load_dicom_block(high_res_files, i, i + high_res_patch_size)
            patch_id = extract_patches(img, patches_path, patch_id, config)

    print(f"Extracted {patch_id} patches from {image_path}")
    return patch_id


def process_batch(images_dir, patches_dir, config=None):
    """Process all image folders in a directory."""
    if config is None:
        config = DEFAULT_CONFIG

    total_patches = 0
    for folder_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, folder_name)
        if not os.path.isdir(image_path):
            continue

        patches_path = os.path.join(patches_dir, f"{folder_name}_patches")
        print(f"Processing: {image_path} -> {patches_path}")

        try:
            count = process_single_image(image_path, patches_path, config)
            total_patches += count
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"Total patches extracted: {total_patches}")
    return total_patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D patches from high-res DICOM images")
    parser.add_argument("input_path", help="Path to image folder (single) or directory of image folders (batch)")
    parser.add_argument("output_path", help="Path to save extracted patches")
    parser.add_argument("--batch", action="store_true", help="Process multiple image folders")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size (default: 64)")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap size (default: 32)")
    parser.add_argument("--threshold", type=float, default=600, help="Intensity threshold (default: 600)")

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["patch_size"] = args.patch_size
    config["overlap_size"] = args.overlap
    config["intensity_threshold"] = args.threshold

    if args.batch:
        process_batch(args.input_path, args.output_path, config)
    else:
        process_single_image(args.input_path, args.output_path, config)
