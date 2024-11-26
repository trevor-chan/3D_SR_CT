import F01_patch_sampling
import F03_patch_sampling
import sys
import os

def run(folder_name, image_folder_path, big_patches_path):
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

        if folder_name == "F03" or folder_name == "F26" or folder_name == "ND17348_new":
            F03_patch_sampling.run(image_folder_path, patches_folder_path)

    except Exception as e:
        print(f"Error processing folder {image_folder_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please pass in <folder name> <image_folder_path> <big_patches_path> as arguments")
        print("Example: python3 F03 /d5/trevor/HRPQCT/linked_combined_toplevel/F03 /1d3/trevor_requires_more_space/3dsrct_patches")
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])