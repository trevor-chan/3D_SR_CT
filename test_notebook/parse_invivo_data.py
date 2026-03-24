import pydicom
import os
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def is_dicom_file(file_path):
    """Check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False


def count_dicom_files(directory):
    """Count the number of DICOM files in a directory."""
    dicom_count = 0
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and is_dicom_file(file_path):
            dicom_count += 1
    return dicom_count


def extract_resolution_metadata(dicom_path):
    """Extract resolution metadata from a DICOM file.
    
    Returns:
        tuple: (xy_resolution, z_step_size) or (None, None) if metadata is missing
    """
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
        # Extract XY resolution (PixelSpacing)
        xy_resolution = None
        if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing is not None:
            # PixelSpacing is typically [row_spacing, column_spacing]
            # For CT scans, these are usually the same, so we'll use the first value
            try:
                # Check if it's a sequence (list, tuple, or MultiValue with length)
                if hasattr(ds.PixelSpacing, '__len__') and len(ds.PixelSpacing) > 0:
                    xy_resolution = float(ds.PixelSpacing[0])
                else:
                    # Single value, convert directly
                    xy_resolution = float(ds.PixelSpacing)
            except (TypeError, ValueError, IndexError) as e:
                # If conversion fails, skip this value
                print(f"Warning: Could not convert PixelSpacing to float: {e}")
                xy_resolution = None
        
        # Extract Z step size (SliceThickness)
        z_step_size = None
        if hasattr(ds, 'SliceThickness') and ds.SliceThickness is not None:
            z_step_size = float(ds.SliceThickness)
        
        return xy_resolution, z_step_size
    
    except Exception as e:
        print(f"Error reading DICOM file {dicom_path}: {e}")
        return None, None


def find_dicom_directories(root_path):
    """Recursively find directories containing multiple DICOM files.
    """
    # First pass: build directory tree and identify leaf directories
    all_dirs = set()
    leaf_dirs = set()
    
    print("Building directory tree...")
    for root, dirs, files in os.walk(root_path):
        all_dirs.add(root)
        # If this directory has no subdirectories, it's a leaf
        if len(dirs) == 0:
            leaf_dirs.add(root)
    
    print(f"Found {len(all_dirs)} total directories, {len(leaf_dirs)} leaf directories")
    
    # Second pass: only check leaf directories for multiple DICOM files
    # (these are the only directories that can have files directly in them)
    print("Checking leaf directories for DICOM files...")
    dicom_dirs = []
    
    for leaf_dir in tqdm(leaf_dirs, desc="Scanning directories"):
        dicom_count = count_dicom_files(leaf_dir)
        if dicom_count > 1:
            dicom_dirs.append(leaf_dir)
    
    return dicom_dirs


def get_first_dicom_file(directory):
    """Get the path to the first DICOM file in a directory."""
    for file in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and is_dicom_file(file_path):
            return file_path
    return None


def main():
    root_directory = "/sdata1/trevor/datasets/Abd_and_Pelvic_CT"
    
    print(f"Searching for DICOM directories in {root_directory}...")
    
    # Find all directories with multiple DICOM files
    dicom_directories = find_dicom_directories(root_directory)
    
    print(f"Found {len(dicom_directories)} directories with multiple DICOM files")
    
    # Extract metadata from each directory
    results = []
    
    for dicom_dir in tqdm(dicom_directories, desc="Processing directories"):
        # Get the first DICOM file in the directory
        first_dicom = get_first_dicom_file(dicom_dir)
        
        if first_dicom is None:
            continue
        
        # Extract resolution metadata
        xy_resolution, z_step_size = extract_resolution_metadata(first_dicom)
        
        # Store results
        results.append({
            'path': dicom_dir,
            'xy_resolution': xy_resolution,
            'z_step_size': z_step_size
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'ct_scan_metadata.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total scans processed: {len(df)}")
    print(f"\nSummary:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    df = main()