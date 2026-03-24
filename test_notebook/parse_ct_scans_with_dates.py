import pydicom
import os
import pandas as pd
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


def extract_acquisition_date(dicom_path):
    """Extract acquisition date from a DICOM file.
    
    Returns:
        str: Acquisition date (YYYYMMDD format) or None if not available
    """
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
        # Try AcquisitionDate first (tag 0008,0022)
        acquisition_date = None
        if hasattr(ds, 'AcquisitionDate') and ds.AcquisitionDate is not None:
            acquisition_date = str(ds.AcquisitionDate).strip()
        # Fall back to StudyDate (tag 0008,0020)
        elif hasattr(ds, 'StudyDate') and ds.StudyDate is not None:
            acquisition_date = str(ds.StudyDate).strip()
        # Fall back to SeriesDate (tag 0008,0021)
        elif hasattr(ds, 'SeriesDate') and ds.SeriesDate is not None:
            acquisition_date = str(ds.SeriesDate).strip()
        
        return acquisition_date
    
    except Exception as e:
        print(f"Error reading DICOM file {dicom_path}: {e}")
        return None


def find_dicom_directories(root_path):
    """Recursively find directories containing multiple DICOM files.
    
    Optimized approach: First builds directory tree, then only checks
    leaf directories (directories with no subdirectories) for DICOM files.
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
    root_directory = "/d2/AI/SI/CT"
    
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
        
        # Extract acquisition date metadata
        acquisition_date = extract_acquisition_date(first_dicom)
        
        # Store results
        results.append({
            'path': dicom_dir,
            'acquisition_date': acquisition_date
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'ct_scan_dates.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total scans processed: {len(df)}")
    print(f"\nDate statistics:")
    print(f"Scans with dates: {df['acquisition_date'].notna().sum()}")
    print(f"Scans without dates: {df['acquisition_date'].isna().sum()}")
    
    return df


if __name__ == "__main__":
    df = main()
