import F01_patch_sampling
import F03_patch_sampling
import sys
import os
from tqdm import tqdm

big_image_folder_path = "/d5/trevor/HRPQCT/linked_combined_toplevel"
big_patches_path = "/1d3/trevor_requires_more_space/3dsrct_patches"
folder_names = ["F03", "F26", "ND17348_new", "0058708-01", "F01"]

def run(folder_names):
    for folder_name in tqdm(folder_names):
        image_folder_path = os.path.join(big_image_folder_path, folder_name)

        if not os.path.isdir(image_folder_path):
            print(f"Image folder path {image_folder_path} does not exist")
            return
        if not os.path.isdir(big_patches_path):
            print(f"Big patches folder path {big_patches_path} does not exist")
            return
        img_name = folder_name + "_patches"
        patches_folder_path = os.path.join(big_patches_path, img_name)
        if os.path.isdir(patches_folder_path):
            os.system(f"rm -rf {patches_folder_path}")
        try:
            os.makedirs(patches_folder_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {patches_folder_path}: {e}")
            return
        try:
            if folder_name == "F01":
                F01_patch_sampling.run(image_folder_path, patches_folder_path)
            else:
                F03_patch_sampling.run(image_folder_path, patches_folder_path)

        except Exception as e:
            print(f"Error processing folder {image_folder_path}: {e}")


if __name__ == "__main__":
    run()
