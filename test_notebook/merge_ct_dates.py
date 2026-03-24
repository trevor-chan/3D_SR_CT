import pandas as pd
import re
from pathlib import Path


def extract_accession_number(path):
    """Extract CT accession number from path.
    
    Pattern: /d2/AI/SI/CT/{accession_number}/...
    Returns: string accession number or None
    """
    if pd.isna(path) or not isinstance(path, str):
        return None
    
    # Pattern: /d2/AI/SI/CT/ followed by accession number, then /
    pattern = r'/d2/AI/SI/CT/([^/]+)'
    match = re.search(pattern, path)
    
    if match:
        return match.group(1)
    return None


def main():
    # Load the master Excel file
    print("Loading master spreadsheet...")
    master_file = 'SI_Master_Trevor.xlsx'
    
    try:
        master_df = pd.read_excel(master_file, engine='openpyxl')
        print(f"Loaded {len(master_df)} rows from {master_file}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    # Check if 'CT Acc #' column exists
    if 'CT Acc #' not in master_df.columns:
        print("Column 'CT Acc #' not found in the master spreadsheet.")
        print(f"Available columns: {master_df.columns.tolist()}")
        return None
    
    # Load the CT scan dates CSV
    print("\nLoading CT scan dates CSV...")
    ct_dates_file = 'ct_scan_dates.csv'
    
    try:
        ct_dates_df = pd.read_csv(ct_dates_file)
        print(f"Loaded {len(ct_dates_df)} rows from {ct_dates_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Extract accession numbers from paths
    print("\nExtracting accession numbers from CT scan paths...")
    ct_dates_df['CT_Acc_Number'] = ct_dates_df['path'].apply(extract_accession_number)
    
    # Check how many were successfully extracted
    extracted_count = ct_dates_df['CT_Acc_Number'].notna().sum()
    print(f"Successfully extracted {extracted_count} accession numbers from {len(ct_dates_df)} paths")
    
    # Prepare CT dates dataframe for merging (keep only relevant columns)
    ct_dates_merge = ct_dates_df[['CT_Acc_Number', 'acquisition_date']].copy()
    ct_dates_merge = ct_dates_merge.rename(columns={'acquisition_date': 'CT_Acquisition_Date'})
    
    # Convert CT Acc # to string for consistent merging (handle any type mismatches)
    master_df['CT Acc #'] = master_df['CT Acc #'].astype(str)
    ct_dates_merge['CT_Acc_Number'] = ct_dates_merge['CT_Acc_Number'].astype(str)
    
    # Perform the merge
    print("\nMerging dataframes...")
    merged_df = master_df.merge(
        ct_dates_merge,
        left_on='CT Acc #',
        right_on='CT_Acc_Number',
        how='left'  # Keep all rows from master, add CT dates where available
    )
    
    # Drop the temporary CT_Acc_Number column (we already have 'CT Acc #')
    merged_df = merged_df.drop(columns=['CT_Acc_Number'])
    
    # Print merge statistics
    print(f"\nMerge Summary:")
    print(f"Total rows in merged dataframe: {len(merged_df)}")
    print(f"Rows with CT acquisition date: {merged_df['CT_Acquisition_Date'].notna().sum()}")
    print(f"Rows without CT acquisition date: {merged_df['CT_Acquisition_Date'].isna().sum()}")
    
    # Save the merged Excel file
    output_file = 'SI_Master_Trevor_With_CT_Dates.xlsx'
    merged_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nMerged data saved to {output_file}")
    
    # Also save as CSV for compatibility
    csv_output = 'SI_Master_Trevor_With_CT_Dates.csv'
    merged_df.to_csv(csv_output, index=False)
    print(f"Merged data also saved to {csv_output}")
    
    return merged_df


if __name__ == "__main__":
    df = main()
