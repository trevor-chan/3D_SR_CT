import pandas as pd
import numpy as np
from datetime import datetime


def parse_date(date_val):
    """Parse various date formats to datetime object.
    
    Returns: datetime object or None
    """
    if pd.isna(date_val):
        return None
    
    # If already a datetime, return it
    if isinstance(date_val, (datetime, pd.Timestamp)):
        return pd.to_datetime(date_val)
    
    # Try to parse string dates
    if isinstance(date_val, str):
        date_val = date_val.strip()
        if not date_val:
            return None
        
        # Try various formats
        formats = [
            '%m/%d/%Y',    # MM/DD/YYYY
            '%Y-%m-%d',    # YYYY-MM-DD
            '%m-%d-%Y',    # MM-DD-YYYY
            '%Y/%m/%d',    # YYYY/MM/DD
            '%d/%m/%Y',    # DD/MM/YYYY
            '%m/%d/%y',    # MM/DD/YY
            '%Y%m%d',      # YYYYMMDD
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_val, fmt)
            except ValueError:
                continue
        
        # Try pandas auto-parsing as last resort
        try:
            return pd.to_datetime(date_val)
        except:
            return None
    
    # Try to convert numbers (Excel serial dates)
    try:
        return pd.to_datetime(date_val)
    except:
        return None


def years_difference(date1, date2):
    """Calculate difference between two dates in years.
    
    Returns: float (positive if date2 > date1, negative otherwise) or None
    """
    if date1 is None or date2 is None:
        return None
    
    try:
        delta = date2 - date1
        # Convert to years (approximate: 365.25 days per year)
        years = delta.days / 365.25
        return round(years, 2)
    except:
        return None


def main():
    # Load the Excel file
    print("Loading Excel file...")
    # Note: Adjust filename if needed (remove ~$ prefix if present)
    excel_file = 'hipfxchartreview_chen_sarah_hurreh (1).xlsx'
    
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
        print(f"Loaded {len(df)} rows from {excel_file}")
    except FileNotFoundError:
        print(f"File not found: {excel_file}")
        print("Please ensure the file exists and update the filename if needed.")
        return None
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    # Print available columns
    print(f"\nAvailable columns: {df.columns.tolist()}")
    
    # Define column names (adjust if column names are slightly different)
    fracture_col = "Date (confrim if fracture date)"
    dxa_col = "DXA Date"
    ct_col = "Date of CT"
    
    # Check if columns exist
    missing_cols = []
    for col in [fracture_col, dxa_col, ct_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"\nWarning: The following columns were not found: {missing_cols}")
        print("Attempting to find similar column names...")
        
        # Try to find similar columns
        for missing in missing_cols:
            for col in df.columns:
                if 'fracture' in col.lower() and 'date' in col.lower():
                    print(f"  Found potential match for fracture date: '{col}'")
                elif 'dxa' in col.lower() and 'date' in col.lower():
                    print(f"  Found potential match for DXA date: '{col}'")
                elif 'ct' in col.lower() and 'date' in col.lower():
                    print(f"  Found potential match for CT date: '{col}'")
        
        print("\nPlease update the column names in the script if needed.")
        return None
    
    # Parse dates
    print("\nParsing dates...")
    df['_fracture_date'] = df[fracture_col].apply(parse_date)
    df['_dxa_date'] = df[dxa_col].apply(parse_date)
    df['_ct_date'] = df[ct_col].apply(parse_date)
    
    # Print parsing statistics
    print(f"Fracture dates parsed: {df['_fracture_date'].notna().sum()} / {len(df)}")
    print(f"DXA dates parsed: {df['_dxa_date'].notna().sum()} / {len(df)}")
    print(f"CT dates parsed: {df['_ct_date'].notna().sum()} / {len(df)}")
    
    # Compute differences in years
    print("\nComputing date differences...")
    
    # DEXA to CT (positive if CT is after DEXA)
    df['Years_DEXA_to_CT'] = df.apply(
        lambda row: years_difference(row['_dxa_date'], row['_ct_date']), axis=1
    )
    
    # CT to Fracture (positive if fracture is after CT)
    df['Years_CT_to_Fracture'] = df.apply(
        lambda row: years_difference(row['_ct_date'], row['_fracture_date']), axis=1
    )
    
    # DEXA to Fracture (positive if fracture is after DEXA)
    df['Years_DEXA_to_Fracture'] = df.apply(
        lambda row: years_difference(row['_dxa_date'], row['_fracture_date']), axis=1
    )
    
    # Compute span from earliest to latest
    def compute_span(row):
        """Compute the span in years from earliest to latest date."""
        dates = [row['_fracture_date'], row['_dxa_date'], row['_ct_date']]
        valid_dates = [d for d in dates if d is not None]
        
        if len(valid_dates) < 2:
            return None
        
        earliest = min(valid_dates)
        latest = max(valid_dates)
        return years_difference(earliest, latest)
    
    df['Years_Earliest_to_Latest'] = df.apply(compute_span, axis=1)
    
    # Drop temporary columns
    df = df.drop(columns=['_fracture_date', '_dxa_date', '_ct_date'])
    
    # Count entries where all three occurred within 1 year
    def all_within_one_year(row):
        """Check if all three dates are within 1 year of each other."""
        span = row['Years_Earliest_to_Latest']
        if span is None:
            return False
        return abs(span) <= 1.0
    
    within_one_year_count = df.apply(all_within_one_year, axis=1).sum()
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Rows with all three dates within 1 year: {within_one_year_count}")
    
    # Print statistics for each computed column
    for col in ['Years_DEXA_to_CT', 'Years_CT_to_Fracture', 'Years_DEXA_to_Fracture', 'Years_Earliest_to_Latest']:
        valid_count = df[col].notna().sum()
        print(f"\n{col}:")
        print(f"  Valid entries: {valid_count}")
        if valid_count > 0:
            print(f"  Mean: {df[col].mean():.2f} years")
            print(f"  Min: {df[col].min():.2f} years")
            print(f"  Max: {df[col].max():.2f} years")
    
    # Save the output
    output_file = 'hipfxchartreview_with_date_diffs.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nOutput saved to {output_file}")
    
    # Also save as CSV
    csv_output = 'hipfxchartreview_with_date_diffs.csv'
    df.to_csv(csv_output, index=False)
    print(f"Output also saved to {csv_output}")
    
    return df


if __name__ == "__main__":
    df = main()
