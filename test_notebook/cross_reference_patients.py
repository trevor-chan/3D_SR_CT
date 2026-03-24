import pandas as pd


def main():
    # Load the spreadsheet with date differences
    print("Loading spreadsheet with date differences...")
    try:
        df = pd.read_excel('hipfxchartreview_with_date_diffs.xlsx', engine='openpyxl')
        print(f"Loaded {len(df)} rows from hipfxchartreview_with_date_diffs.xlsx")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        # Try CSV fallback
        try:
            df = pd.read_csv('hipfxchartreview_with_date_diffs.csv')
            print(f"Loaded {len(df)} rows from hipfxchartreview_with_date_diffs.csv")
        except Exception as e2:
            print(f"Error loading CSV file: {e2}")
            return None
    
    # Load the matching patient IDs
    print("\nLoading matching patient IDs...")
    try:
        with open('matching_patient_ids.txt', 'r') as f:
            matching_ids = set()
            for line in f:
                line = line.strip()
                if line:
                    matching_ids.add(line)
        print(f"Loaded {len(matching_ids)} patient IDs from matching_patient_ids.txt")
    except Exception as e:
        print(f"Error loading patient IDs: {e}")
        return None
    
    # Find CT accession number column
    ct_acc_col = 'Accession number of CT'
    if ct_acc_col not in df.columns:
        print(f"Column '{ct_acc_col}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Filter for rows where ALL THREE dates are present and within 1 year
    # Require all three pairwise differences to be non-nan (ensures all 3 dates exist)
    print("\nFiltering for entries where all THREE dates are present and within 1 year...")
    
    has_all_three_dates = (
        df['Years_DEXA_to_CT'].notna() & 
        df['Years_CT_to_Fracture'].notna() & 
        df['Years_DEXA_to_Fracture'].notna()
    )
    
    within_one_year = df[
        has_all_three_dates & 
        df['Years_Earliest_to_Latest'].notna() & 
        (df['Years_Earliest_to_Latest'].abs() <= 1.0)
    ]
    
    # Also count entries that would have matched but are missing dates
    missing_dates = df[
        ~has_all_three_dates & 
        df['Years_Earliest_to_Latest'].notna() & 
        (df['Years_Earliest_to_Latest'].abs() <= 1.0)
    ]
    
    print(f"Entries with all THREE dates within 1 year: {len(within_one_year)}")
    print(f"Entries within 1 year but MISSING one or more dates: {len(missing_dates)}")
    
    # Extract CT accession numbers from filtered entries
    ct_accession_numbers = set()
    for acc in within_one_year[ct_acc_col]:
        if pd.notna(acc):
            # Convert to string and clean
            acc_str = str(acc).strip()
            # Remove any non-numeric characters for comparison
            acc_numeric = ''.join(c for c in acc_str if c.isdigit())
            if acc_numeric:
                ct_accession_numbers.add(acc_numeric)
            # Also add original for matching
            if acc_str:
                ct_accession_numbers.add(acc_str)
    
    print(f"Unique CT accession numbers from within-1-year entries: {len(ct_accession_numbers)}")
    
    # Cross-reference with matching patient IDs
    print("\nCross-referencing with matching patient IDs...")
    matching_both = ct_accession_numbers.intersection(matching_ids)
    
    print(f"\n=== Results ===")
    print(f"Total entries with all dates within 1 year: {len(within_one_year)}")
    print(f"CT accession numbers from those entries: {len(ct_accession_numbers)}")
    print(f"Patient IDs in matching_patient_ids.txt: {len(matching_ids)}")
    print(f"\n*** Patients meeting BOTH criteria: {len(matching_both)} ***")
    
    if matching_both:
        print(f"\nMatching patient IDs:")
        for pid in sorted(matching_both):
            print(f"  {pid}")
        
        # Save to file
        output_file = 'patients_meeting_both_criteria.txt'
        with open(output_file, 'w') as f:
            for pid in sorted(matching_both):
                f.write(f"{pid}\n")
        print(f"\nList saved to {output_file}")
    
    # Also show the full data for these patients
    if matching_both:
        print("\n=== Details for matching patients ===")
        for pid in sorted(matching_both):
            patient_rows = within_one_year[
                within_one_year[ct_acc_col].astype(str).str.contains(pid, na=False)
            ]
            if len(patient_rows) > 0:
                row = patient_rows.iloc[0]
                print(f"\nPatient ID: {pid}")
                print(f"  Years_DEXA_to_CT: {row.get('Years_DEXA_to_CT', 'N/A')}")
                print(f"  Years_CT_to_Fracture: {row.get('Years_CT_to_Fracture', 'N/A')}")
                print(f"  Years_DEXA_to_Fracture: {row.get('Years_DEXA_to_Fracture', 'N/A')}")
                print(f"  Years_Earliest_to_Latest: {row.get('Years_Earliest_to_Latest', 'N/A')}")
    
    return matching_both


if __name__ == "__main__":
    result = main()
