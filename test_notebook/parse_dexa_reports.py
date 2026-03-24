import pandas as pd
import re
from tqdm import tqdm


def extract_age(dexa_text):
    """Extract age from DEXA report text.
    
    Pattern: "Age: 38.9," or "Age: 38.9"
    Returns: float or None
    """
    if pd.isna(dexa_text) or not isinstance(dexa_text, str):
        return None
    
    # Pattern: Age: followed by optional whitespace, then a float, then optional comma
    pattern = r'Age:\s*([0-9]+\.?[0-9]*)'
    match = re.search(pattern, dexa_text)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_scan_date(dexa_text):
    """Extract scan date from DEXA report text.
    
    Pattern: "Date of Scan: 10/20/2020"
    Returns: string (MM/DD/YYYY format) or None
    """
    if pd.isna(dexa_text) or not isinstance(dexa_text, str):
        return None
    
    # Pattern: Date of Scan: followed by optional whitespace, then MM/DD/YYYY
    pattern = r'Date of Scan:\s*(\d{1,2}/\d{1,2}/\d{4})'
    match = re.search(pattern, dexa_text)
    
    if match:
        return match.group(1)
    return None


def main():
    # Load the Excel file
    print("Loading Excel file...")
    excel_file = 'SI_Master.xlsx'
    
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} rows from {excel_file}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    # Check if column 'DEXAReport' exists
    if 'DEXAReport' not in df.columns:
        print("Column 'DEXAReport' not found in the Excel file.")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    print(f"Found {df['DEXAReport'].notna().sum()} non-null entries in column 'DEXAReport'")
    
    # Extract age and scan date from DEXA reports
    print("\nExtracting age and scan date from DEXA reports...")
    
    ages = []
    scan_dates = []
    
    for idx, dexa_text in tqdm(df['DEXAReport'].items(), total=len(df), desc="Processing reports"):
        age = extract_age(dexa_text)
        scan_date = extract_scan_date(dexa_text)
        
        ages.append(age)
        scan_dates.append(scan_date)
    
    # Add extracted data to dataframe with specified column names
    df['AgeAtScan'] = ages
    df['ScanDate'] = scan_dates
    
    # Create summary statistics
    print(f"\nExtraction Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with age extracted: {df['AgeAtScan'].notna().sum()}")
    print(f"Rows with scan date extracted: {df['ScanDate'].notna().sum()}")
    print(f"Rows with both age and scan date: {(df['AgeAtScan'].notna() & df['ScanDate'].notna()).sum()}")
    
    # Save Excel file with original data plus new columns
    excel_output = 'SI_Master_Trevor.xlsx'
    df.to_excel(excel_output, index=False, engine='openpyxl')
    print(f"\nExcel file with added columns saved to {excel_output}")
    
    # Also save CSV files for compatibility
    output_file = 'dexa_extracted_data.csv'
    df.to_csv(output_file, index=False)
    print(f"CSV results saved to {output_file}")
    
    # Also save just the extracted columns for easier viewing
    extracted_df = df[['AgeAtScan', 'ScanDate']].copy()
    extracted_output = 'dexa_extracted_summary.csv'
    extracted_df.to_csv(extracted_output, index=False)
    print(f"Summary (AgeAtScan and ScanDate only) saved to {extracted_output}")
    
    return df


if __name__ == "__main__":
    df = main()
