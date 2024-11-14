import os
import patch_sampling
import sys

# def run(big_img_folder_path, save_patches_path):
#     for p in os.listdir(big_img_folder_path):
#         img_folder_path = os.path.join(big_img_folder_path, p)
#         # print(img_folder_path)
#         # print(p)
#         img_name = p.split("/")[-1] + "_patches"
#         patches_folder_path = save_patches_path + '/' + img_name
#         print(save_patches_path)
#         print(patches_folder_path)
#         os.mkdir(patches_folder_path)
#         patch_sampling.run(img_folder_path, patches_folder_path)


def run(big_img_folder_path, save_patches_path):
    for p in os.listdir(big_img_folder_path):
        img_folder_path = os.path.join(big_img_folder_path, p)
        if not os.path.isdir(img_folder_path):
            print(f"Skipping non-directory: {img_folder_path}")
            continue
        img_name = p + "_patches"
        patches_folder_path = os.path.join(save_patches_path, img_name)
        print(f"Image folder path: {img_folder_path}")
        print(f"Patches folder path: {patches_folder_path}")
        try:
            os.makedirs(patches_folder_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {patches_folder_path}: {e}")
            continue
        try:
            patch_sampling.run(img_folder_path, patches_folder_path)
        except Exception as e:
            print(f"Error processing folder {img_folder_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please pass in <big_image_path> <big_patches_path> as arguments")
        print("Example: python final_patch_sampling.py /path/to/high_res_images_folder /path/to/save_patches_folder")
    else:
        run(sys.argv[1], sys.argv[2])