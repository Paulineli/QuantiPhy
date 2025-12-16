import pandas as pd
import glob
import os
import argparse
import sys

"""
This script is used to calculate the MRA metrics for the QuantiPhy dataset.

Usage:
python evaluator.py <input_dir> <output_dir> [--gt_file GT_FILE]

Example:
python evaluator.py /Users/paulineli/Desktop/CIB/QuantiPhy/additional_tasks/original /Users/paulineli/Desktop/CIB/QuantiPhy/mra_experiments/original
Args:
    input_dir: Input directory containing CSV files to process
    output_dir: Output directory to save processed results
    gt_file: Optional ground truth CSV file (default: quantiphy_validation.csv in current directory)

Returns:
    A CSV file containing the MRA metrics for each model and the average MRA for each category and background class
"""
parser = argparse.ArgumentParser(description='Process CSV files and calculate MRA metrics.')
parser.add_argument('input_dir', type=str, help='Input directory containing CSV files to process')
parser.add_argument('output_dir', type=str, help='Output directory to save processed results')
parser.add_argument('--gt_file', type=str, default='quantiphy_validation.csv', 
                    help='Ground truth CSV file (default: quantiphy_validation.csv in current directory)')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
gt_file = args.gt_file

os.makedirs(output_dir, exist_ok=True)

# Load and validate ground truth file if provided
gt_df = None
if gt_file and os.path.exists(gt_file):
    print(f"Loading ground truth file: {gt_file}")
    gt_df = pd.read_csv(gt_file)
    
    # Get the first column name
    first_col = gt_df.columns[0]
    
    # Check if first column is integer
    # try:
    #     gt_df[first_col] = pd.to_numeric(gt_df[first_col], errors='raise').astype(int)
    #     print(f"  ✓ First column '{first_col}' contains integer IDs")
    # except (ValueError, TypeError) as e:
    #     raise ValueError(f"First column '{first_col}' in GT file must contain integer values. Error: {e}")
    
    # Check if 'ground_truth_posterior' column exists
    if 'ground_truth_posterior' not in gt_df.columns:
        raise ValueError(f"Column 'ground_truth_posterior' not found in GT file. Available columns: {list(gt_df.columns)}")
    print(f"  ✓ Column 'ground_truth_posterior' exists")
    
    # Check if all values in 'ground_truth_posterior' are float
    gt_posterior_numeric = pd.to_numeric(gt_df['ground_truth_posterior'], errors='coerce')
    non_float_count = gt_df['ground_truth_posterior'].notna().sum() - gt_posterior_numeric.notna().sum()
    if non_float_count > 0:
        raise ValueError(f"Found {non_float_count} non-float values in 'ground_truth_posterior' column")
    print(f"  ✓ All values in 'ground_truth_posterior' are float")
    
    # Create a mapping from ID to ground_truth_posterior
    gt_mapping = dict(zip(gt_df[first_col], gt_posterior_numeric))
    print(f"  ✓ Loaded {len(gt_mapping)} ground truth entries\n")
elif gt_file:
    print(f"Warning: GT file '{gt_file}' not found. Using ground_truth_posterior from input files.\n")

csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

if len(csv_files) == 0:
    print(f"Warning: No CSV files found in input directory: {input_dir}")
    print("Creating empty results file.")
    results_df = pd.DataFrame()
    summary_output_path = os.path.join(output_dir, 'all_model_results.csv')
    results_df.to_csv(summary_output_path, index=False)
    print(f"Empty summary saved to: {summary_output_path}")
    sys.exit(0)

print(f"Found {len(csv_files)} CSV file(s) to process\n")

all_results = []

for input_file in csv_files:
    file_name = os.path.basename(input_file)
    model_name = file_name.replace('merged_', '').replace('.csv', '')
    print(f"\nprocessing: {file_name}")
    
    result_row = {'model': model_name}
    
    df = pd.read_csv(input_file)
    
    # Change last digit of video_type to 'X' if video_source is 'segmentation'
    if 'video_source' in df.columns:
        mask = df['video_source'] == 'segmentation'
        df.loc[mask, 'video_type'] = df.loc[mask, 'video_type'].apply(
            lambda x: x[:-1] + 'X' if pd.notna(x) and len(x) > 0 else x
        )
    
    df = df.sort_values(by=['video_id', 'question'])
    
    # Replace ground_truth_posterior with values from GT file if available
    if gt_df is not None:
        # Get the ID column from input file (first column)
        input_id_col = df.columns[0]
        
        # Convert to integer for matching (handle NaN values)
        try:
            input_ids_numeric = pd.to_numeric(df[input_id_col], errors='coerce')
            # Convert to Int64 (nullable integer type) to preserve NaN values
            input_ids = input_ids_numeric.astype('Int64')
            df[input_id_col] = input_ids
        except (ValueError, TypeError) as e:
            print(f"  Warning: Could not convert first column '{input_id_col}' to integer. Error: {e}. Skipping GT replacement.")
        else:
            # Replace ground_truth_posterior with values from GT mapping
            # Use the integer IDs to map (pandas map handles Int64 with NaN correctly)
            df['ground_truth_posterior'] = input_ids.map(gt_mapping)
            matched_count = df['ground_truth_posterior'].notna().sum()
            print(f"  Matched {matched_count}/{len(df)} rows with GT file")
    
    # Calculate percentage of invalid parsed_value records
    total_records = len(df)
    blank_or_invalid = df['parsed_value'].isna().sum()
    
    # Convert to numeric to identify non-numerical values
    parsed_numeric = pd.to_numeric(df['parsed_value'], errors='coerce')
    non_numerical = (df['parsed_value'].notna() & parsed_numeric.isna()).sum()
    zero_values = (parsed_numeric == 0).sum()
    
    invalid_count = blank_or_invalid + non_numerical + zero_values
    invalid_percentage = (invalid_count / total_records * 100) if total_records > 0 else 0
    
    result_row['invalid_percentage'] = invalid_percentage
    
    print(f"  invalid parsed_value records: {invalid_count}/{total_records} ({invalid_percentage:.2f}%)")
    print(f"    - blank: {blank_or_invalid}")
    print(f"    - non-numerical: {non_numerical}")
    print(f"    - zero: {zero_values}")
    
    df['category'] = df.apply(lambda row: row['inference_type'][0] + row['video_type'][1] 
                              if pd.notna(row['video_type']) else None, axis=1)
    
    df['parsed_value'] = pd.to_numeric(df['parsed_value'], errors='coerce').abs()
    df['ground_truth_posterior'] = pd.to_numeric(df['ground_truth_posterior'], errors='coerce')
    
    df['bg'] = df['video_type'].apply(lambda x: x[-1] if pd.notna(x) and len(x) > 0 else None)
    df['obj'] = df['video_type'].apply(lambda x: 'single' if pd.notna(x) and len(x) > 2 and x[2] == 'S' 
                                        else 'multiple' if pd.notna(x) and len(x) > 2 and x[2] == 'M' 
                                        else None)
    
    C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    df['mra'] = df.apply(lambda row: sum(
        abs(row['parsed_value'] - row['ground_truth_posterior']) / row['ground_truth_posterior'] < (1 - theta)
        for theta in C
    ) / 10 if pd.notna(row['ground_truth_posterior']) and row['ground_truth_posterior'] != 0 else float('nan'), axis=1)
    
    mra = {}
    for category in df['category'].unique():
        if pd.notna(category):
            category_df = df[df['category'] == category]
            mra[category] = category_df['mra'].mean()
            print(f"  average mra for {category}: {mra[category]:.4f}")
    
    # Store category MRAs
    for category in ['S2', 'D2', 'S3', 'D3']:
        result_row[f'mra_{category}'] = mra.get(category, None)
    
    if 'S2' in mra and 'D2' in mra and 'S3' in mra and 'D3' in mra:
        average_mra = (mra['S2'] + mra['D2'] + mra['S3'] + mra['D3']) / 4
        result_row['mra_average'] = average_mra
        print(f"  average mra over four categories: {average_mra:.4f}")
    else:
        result_row['mra_average'] = None
    
    # Calculate 4-category average MRA for each bg class
    print("\n  MRA by background (bg) class:")
    for bg_value in sorted(df['bg'].dropna().unique()):
        bg_df = df[df['bg'] == bg_value]
        bg_mra = {}
        for category in ['S2', 'D2', 'S3', 'D3']:
            category_df = bg_df[bg_df['category'] == category]
            if len(category_df) > 0:
                bg_mra[category] = category_df['mra'].mean()
        
        if len(bg_mra) == 4:
            bg_average = (bg_mra['S2'] + bg_mra['D2'] + bg_mra['S3'] + bg_mra['D3']) / 4
            result_row[f'mra_bg_{bg_value}'] = bg_average
            print(f"    bg={bg_value}: {bg_average:.4f}")
        else:
            result_row[f'mra_bg_{bg_value}'] = None
    
    # Calculate 4-category average MRA for each obj class
    print("\n  MRA by object (obj) class:")
    for obj_value in sorted(df['obj'].dropna().unique()):
        obj_df = df[df['obj'] == obj_value]
        obj_mra = {}
        for category in ['S2', 'D2', 'S3', 'D3']:
            category_df = obj_df[obj_df['category'] == category]
            if len(category_df) > 0:
                obj_mra[category] = category_df['mra'].mean()
        
        if len(obj_mra) == 4:
            obj_average = (obj_mra['S2'] + obj_mra['D2'] + obj_mra['S3'] + obj_mra['D3']) / 4
            result_row[f'mra_obj_{obj_value}'] = obj_average
            print(f"    obj={obj_value}: {obj_average:.4f}")
        else:
            result_row[f'mra_obj_{obj_value}'] = None
    
    all_results.append(result_row)

# Save all results to CSV
if len(all_results) == 0:
    print("\n\nWarning: No results to save. No CSV files were successfully processed.")
    results_df = pd.DataFrame()
else:
    results_df = pd.DataFrame(all_results)
    print(f"\n\nProcessed {len(all_results)} model(s)")

summary_output_path = os.path.join(output_dir, 'all_model_results.csv')
results_df.to_csv(summary_output_path, index=False)
print(f"Summary saved to: {summary_output_path}")
