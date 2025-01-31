import os
import numpy as np
import pydicom
from scipy import ndimage
import tqdm
import sys

# config
high_res_voxel_size = 0.033
desired_voxel_size = 0.2
patch_size = 64
overlap_size = 32


def run(high_res_imge_path, save_patches_path):

    def get_sorted_dicom_files(path):
        dicom_files = [os.path.join(path, f) for f in os.listdir(path)]
        return sorted(dicom_files)

    def load_dicom_block(dicom_files, start, end):
        dicom_slices = [pydicom.dcmread(f).pixel_array for f in dicom_files[start:end]]
        img = np.stack(dicom_slices, axis=-1)
        return img

    def extract_patches(img, patch_id):
        downsampled_img = ndimage.zoom(img, high_res_voxel_size / desired_voxel_size)
        x_size, y_size, z_size = downsampled_img.shape
        if z_size < patch_size:
            print(f"z size is less than patch size: {z_size}")
            return patch_id
        else:
            downsampled_img = downsampled_img[:, :, :patch_size]

        for x in range(0, x_size, patch_size - overlap_size):
            for y in range(0, y_size, patch_size - overlap_size):
                if x + patch_size <= x_size and y + patch_size <= y_size:
                    patch = downsampled_img[x : x + patch_size, y : y + patch_size, :]
                    avg_intensity = np.mean(patch)
                    if avg_intensity > 600:
                        patch_filename = os.path.join(
                            patches_path, f"{str(patch_id).zfill(8)}.npy"
                        )
                        np.save(patch_filename, patch)
                        patch_id += 1

        return patch_id

    high_res_path = high_res_imge_path
    patches_path = save_patches_path

    high_res_files = get_sorted_dicom_files(high_res_path)
    high_res_patch_size = int(patch_size * desired_voxel_size / high_res_voxel_size + 1)
    high_res_overlap_size = int(
        overlap_size * desired_voxel_size / high_res_voxel_size + 1
    )
    patch_id = 0

    for i in tqdm.tqdm(
        range(0, len(high_res_files), high_res_patch_size - high_res_overlap_size)
    ):
        if i + high_res_patch_size <= len(high_res_files):
            img = load_dicom_block(high_res_files, i, i + high_res_patch_size)
            patch_id = extract_patches(img, patch_id)
            print(f"Processed {i} of {len(high_res_files)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please pass in <high_res_image_path> <save_patches_path> as arguments")
        print("Example: python patch_sampling.py /path/to/high_res_images /path/to/save_patches")
    else:
        run(sys.argv[1], sys.argv[2])